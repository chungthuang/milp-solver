pub mod error;

use std::ops::{AddAssign, Mul, Sub};

use crate::error::Error;
use good_lp::solvers::lp_solvers::LpSolution;
use good_lp::{
    solvers::lp_solvers::{CbcSolver, LpSolver},
    variable, Constraint, Expression, IntoAffineExpression, ProblemVariables, Solution,
    SolverModel, Variable,
};
use log::debug;
use parachain_client::{
    AcceptedProduct, FlexibleProduct, MarketSolution, ProductId, SelectedFlexibleLoad,
};
use uuid::Uuid;

const DECISION_ACCEPTED: f64 = 1.;
// Used in Big-M constraint for selecting only 1 flexible product
const PERCENTAGE_UPPER_BOUND: f64 = 100.;

#[derive(Copy, Clone)]
enum ProductType {
    Bid,
    Offer,
}

pub fn solve(
    bids: Vec<(ProductId, FlexibleProduct)>,
    offers: Vec<(ProductId, FlexibleProduct)>,
    periods: u32,
) -> Result<MarketSolution, Error> {
    // Indicators for flexible load constraint and percentage of a product that's accepted
    let mut vars = ProblemVariables::new();

    let bid_formulation = formulate(&mut vars, &bids, periods as usize);
    let offer_formulation = formulate(&mut vars, &offers, periods as usize);

    let social_welfare: Expression = bid_formulation.welfare.sub(offer_formulation.welfare);
    debug!("social welfare {:?}", social_welfare);

    let problem_id = Uuid::new_v4();
    debug!("Problem ID {:?}", problem_id);
    let cbc_solver =
        LpSolver(CbcSolver::new().with_temp_solution_file(format!("/tmp/milp_{}.sol", problem_id)));

    let mut model = vars.maximise(social_welfare).using(cbc_solver);

    for (period, (bids_quantities, asks_quantities)) in bid_formulation
        .quantities
        .into_iter()
        .zip(offer_formulation.quantities)
        .enumerate()
    {
        let quantity_match = bids_quantities.sub(asks_quantities);
        debug!(
            "Period {}: quantity match constraint {:?}",
            period, quantity_match
        );
        model = model.with(quantity_match.eq(0));
    }

    for bid in bid_formulation.constraints.into_iter() {
        model = model.with(bid.sum_to_one_constraint);
        for big_m in bid.big_m_constraints {
            model = model.with(big_m);
        }
    }

    for offer in offer_formulation.constraints.into_iter() {
        model = model.with(offer.sum_to_one_constraint);
        for big_m in offer.big_m_constraints {
            model = model.with(big_m);
        }
    }

    let sol = model
        .solve()
        .map_err(|err| Error::Solver(err.to_string()))?;

    evaluate(
        sol,
        bids,
        offers,
        bid_formulation.vars,
        offer_formulation.vars,
        periods as usize,
    )
}

struct ProblemFormulation {
    vars: Vec<FlexibleProductVars>,
    constraints: Vec<FlexibleProductConstraints>,
    // A vector of bid/ask quantities for each period
    quantities: Vec<Expression>,
    // Utility/cost
    welfare: Expression,
}

struct FlexibleProductVars {
    // binary variable that decides which load is accepted
    selected_flex_load: Vec<Variable>,
    // What percentage is accepted
    percentage: Vec<Variable>,
}

struct FlexibleProductConstraints {
    // Only load can be accepted
    sum_to_one_constraint: Constraint,
    // Big-M makes sure selected_flex_load = 1 when percentage > 0
    // https://docs.mosek.com/modeling-cookbook/mio.html#integer-modeling
    big_m_constraints: Vec<Constraint>,
}

/// Add decision variables for products to vars, return the decision variables
fn formulate(
    vars: &mut ProblemVariables,
    products: &[(ProductId, FlexibleProduct)],
    periods: usize,
) -> ProblemFormulation {
    let mut product_vars = Vec::with_capacity(products.len());
    let mut product_constraints = Vec::with_capacity(products.len());
    let mut quantities: Vec<Expression> = Vec::with_capacity(periods as usize);
    for _ in 0..periods {
        quantities.push(Expression::default());
    }
    let mut welfare = Expression::default();
    for (_id, flexible_product) in products.iter() {
        let (per_product_vars, per_product_constraints) =
            formulate_per_product(vars, &mut quantities, &mut welfare, flexible_product);
        product_vars.push(per_product_vars);
        product_constraints.push(per_product_constraints);
    }
    ProblemFormulation {
        vars: product_vars,
        constraints: product_constraints,
        quantities,
        welfare,
    }
}

fn formulate_per_product(
    vars: &mut ProblemVariables,
    quantities_by_period: &mut Vec<Expression>,
    welfare: &mut Expression,
    flex_product: &FlexibleProduct,
) -> (FlexibleProductVars, FlexibleProductConstraints) {
    let selected_flex_load = vars.add_vector(variable().binary(), flex_product.len());
    let sum_to_one_constraint = selected_flex_load.iter().sum::<Expression>().leq(1);
    // Per product in flexible
    let mut accepted_perct: Vec<Variable> = Vec::new();
    let mut big_m_constraints: Vec<Constraint> = Vec::new();

    for (product, selected) in flex_product.iter().zip(selected_flex_load.iter()) {
        let perct = if product.can_partially_accept {
            vars.add(variable().clamp(0, 1))
        } else {
            vars.add(variable().binary())
        };
        big_m_constraints.push(
            perct
                .into_expression()
                .leq(selected.mul(PERCENTAGE_UPPER_BOUND)),
        );

        for period in product.start_period..product.end_period {
            accepted_perct.push(perct);

            let quantity = perct.mul(product.quantity as f64);
            quantities_by_period[period as usize].add_assign(quantity.clone());
            welfare.add_assign(quantity.mul(product.price as f64));
        }
    }
    (
        FlexibleProductVars {
            selected_flex_load,
            percentage: accepted_perct,
        },
        FlexibleProductConstraints {
            sum_to_one_constraint,
            big_m_constraints,
        },
    )
}

fn evaluate(
    sol: LpSolution,
    bids: Vec<(ProductId, FlexibleProduct)>,
    offers: Vec<(ProductId, FlexibleProduct)>,
    bid_vars: Vec<FlexibleProductVars>,
    offer_vars: Vec<FlexibleProductVars>,
    periods: usize,
) -> Result<MarketSolution, Error> {
    let bids_solutions =
        evaluate_product_decisions(&sol, bids, bid_vars, ProductType::Bid, periods)?;
    let asks_solutions =
        evaluate_product_decisions(&sol, offers, offer_vars, ProductType::Offer, periods)?;

    let mut auction_prices = Vec::with_capacity(periods);
    for (period, (min_bid_price, max_ask_price)) in bids_solutions
        .auction_prices
        .iter()
        .zip(asks_solutions.auction_prices)
        .enumerate()
    {
        debug!(
            "Period {}: Min bid price {}, max ask price at period {}",
            period, min_bid_price, max_ask_price
        );
        if *min_bid_price < max_ask_price {
            return Err(Error::InvalidSolution(format!(
                "Period {}: Minimum bid price {} is less than maximum ask price {}",
                period, min_bid_price, max_ask_price
            )));
        }
        auction_prices.push(*min_bid_price);
    }

    for (period, (bid_quantity, ask_quantity)) in bids_solutions
        .quantities
        .iter()
        .zip(asks_solutions.quantities)
        .enumerate()
    {
        if *bid_quantity != ask_quantity {
            return Err(Error::InvalidSolution(format!(
                "Period {}: Total bid quantity {} != total ask quantity {}",
                period, bid_quantity, ask_quantity
            )));
        }
    }

    Ok(MarketSolution {
        bids: bids_solutions.solutions,
        asks: asks_solutions.solutions,
        auction_prices,
    })
}

struct ProductDecisions {
    solutions: Vec<AcceptedProduct>,
    // Auction price by period
    auction_prices: Vec<u64>,
    // Quantity by period
    quantities: Vec<u64>,
}

/// Decide if a product is accepted, and if so, which flexible load
fn evaluate_product_decisions(
    sol: &LpSolution,
    products: Vec<(ProductId, FlexibleProduct)>,
    decisions: Vec<FlexibleProductVars>,
    product_type: ProductType,
    periods: usize,
) -> Result<ProductDecisions, Error> {
    let mut solutions: Vec<AcceptedProduct> = Vec::with_capacity(products.len());
    let mut auction_prices: Vec<u64> = Vec::with_capacity(periods);
    let mut quantities: Vec<u64> = Vec::with_capacity(periods);
    for _ in 0..periods {
        auction_prices.push(0);
        quantities.push(0);
    }

    assert_eq!(products.len(), decisions.len());
    for ((id, flex_product), decisions) in products.into_iter().zip(decisions) {
        assert_eq!(flex_product.len(), decisions.selected_flex_load.len());
        assert_eq!(flex_product.len(), decisions.percentage.len());

        let mut accepted_schedule: Option<SelectedFlexibleLoad> = None;
        for schedule_index in 0..flex_product.len() {
            let product = flex_product[schedule_index];
            let accepted = sol.value(decisions.selected_flex_load[schedule_index]);
            let perct = sol.value(decisions.percentage[schedule_index]);

            if accepted != DECISION_ACCEPTED {
                assert_eq!(perct, 0.);
                continue;
            }

            // Make sure at most only 1 flexible load is accepted
            if accepted_schedule.is_some() {
                return Err(Error::InvalidSolution(
                    "Multiple flexible load accepted".to_owned(),
                ));
            }
            let accepted_product = AcceptedProduct {
                id,
                load_index: schedule_index as SelectedFlexibleLoad,
                percentage: (perct * 100.).round() as u8,
            };
            accepted_schedule = Some(accepted_product.load_index);
            solutions.push(accepted_product);

            for period in product.start_period..product.end_period {
                let period = period as usize;
                if auction_prices[period] == 0 {
                    auction_prices[period] = product.price;
                } else {
                    match product_type {
                        ProductType::Bid => {
                            if product.price < auction_prices[period] {
                                auction_prices[period] = product.price;
                            }
                        }
                        ProductType::Offer => {
                            if product.price > auction_prices[period] {
                                auction_prices[period] = product.price;
                            }
                        }
                    };
                }
                quantities[period] += (product.quantity as f64 * perct).round() as u64;
            }
        }
    }

    Ok(ProductDecisions {
        solutions,
        auction_prices,
        quantities,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use parachain_client::Product;
    use test_log::test;

    // Evaluate solution for single period products
    #[test]
    fn test_solve_single_products() {
        let bid_1_price = 6;
        let bid_1 = (
            product_id(1),
            fixed_load(Product {
                price: bid_1_price,
                quantity: 5,
                start_period: 1,
                end_period: 2,
            }),
        );
        let ask_1 = (
            product_id(2),
            fixed_load(Product {
                price: 5,
                quantity: 5,
                start_period: 1,
                end_period: 2,
            }),
        );

        let bid_2_price = 8;
        let bid_2 = (
            product_id(3),
            fixed_load(Product {
                price: bid_2_price,
                quantity: 2,
                start_period: 3,
                end_period: 4,
            }),
        );
        let ask_2 = (
            product_id(4),
            fixed_load(Product {
                price: 7,
                quantity: 2,
                start_period: 3,
                end_period: 4,
            }),
        );

        let bids = vec![bid_1.clone(), bid_2.clone()];

        let asks = vec![ask_1.clone(), ask_2.clone()];

        let solution = solve(bids, asks, 5).unwrap();
        assert_eq!(
            solution.bids,
            vec![
                (bid_1.0, selected_load_index(0)),
                (bid_2.0, selected_load_index(0))
            ]
        );
        assert_eq!(
            solution.asks,
            vec![
                (ask_1.0, selected_load_index(0)),
                (ask_2.0, selected_load_index(0)),
            ]
        );
        assert_eq!(
            solution.auction_prices,
            vec![0, bid_1_price, 0, bid_2_price, 0]
        )
    }

    #[test]
    fn test_solve_single_products_max_social_welfare() {
        let bid_1 = (
            product_id(1),
            fixed_load(Product {
                price: 6,
                quantity: 5,
                start_period: 1,
                end_period: 2,
            }),
        );
        let ask_1 = (
            product_id(2),
            fixed_load(Product {
                price: 6,
                quantity: 5,
                start_period: 1,
                end_period: 2,
            }),
        );

        let bid_2_price = 7;
        let bid_2 = (
            product_id(3),
            fixed_load(Product {
                price: bid_2_price,
                quantity: 5,
                start_period: 1,
                end_period: 2,
            }),
        );
        let ask_2 = (
            product_id(4),
            fixed_load(Product {
                price: 7,
                quantity: 5,
                start_period: 1,
                end_period: 2,
            }),
        );

        let bids = vec![bid_1, bid_2.clone()];

        let asks = vec![ask_1.clone(), ask_2];

        // bid_2 will match with ask_1, because the social welfare would be 7 * 5 - 6 * 5 = 5
        // if bid_1 matches with ask_1 and bid_2 matches with ask_2, than the social welfare would be 0
        let solution = solve(bids, asks, 3).unwrap();
        assert_eq!(solution.bids, vec![(bid_2.0, selected_load_index(0))]);
        assert_eq!(solution.asks, vec![(ask_1.0, selected_load_index(0))]);
        assert_eq!(solution.auction_prices, vec![0, bid_2_price, 0])
    }
    /*
    #[test]
    fn test_lp_temporal_constraint() {
        // Ask price & quantity. 2 periods
        let ap1 = 5;
        let aq1 = 3;

        let ap2 = 3;
        let aq2 = 3;

        // Bid price & quantity. 2 Periods
        let bp1 = 8;
        let bq1 = 2;

        let bp2 = 6;
        let bq2 = 3;

        let bp3 = 7;
        let bq3 = 2;

        variables! {
            vars:
                0 <= a1 <= 1;
                0 <= a2 <= 1;
                0 <= b1 <= 1;
                0 <= b2 <= 1;
                0 <= b3 <= 1;
        } // variables can also be added dynamically
        let cbc_solver =
            LpSolver(CbcSolver::new().with_temp_solution_file("/tmp/test_lp.sol".to_owned()));
        let social_welfare = 2 * bp1 * bq1 * b1 + bp2 * bq2 * b2 + b1 + bp3 * bq3 * b3
            - 2 * ap1 * aq1 * a1
            - ap2 * aq2 * a2;
        let solution = vars
            .maximise(&social_welfare)
            .using(cbc_solver)
            // first period
            .with(constraint!((bq1 * b1 + bq2 * b2) == (aq1 * a1 + aq2 * a2)))
            // second period
            .with(constraint!((bq1 * b1 + bq3 * b3) == (aq1 * a1)))
            .solve()
            .unwrap();
        println!(
            "a1={} a2={} b1={} b2={}, b3={}",
            solution.value(a1),
            solution.value(a2),
            solution.value(b1),
            solution.value(b2),
            solution.value(b3)
        );
        println!("Social welfare = {}", solution.eval(social_welfare));
        assert!(false);
    }

    #[test]
    fn test_lp_compare_double_auction() {
        // Ask price & quantity
        let ap1 = 5;
        let aq1 = 3;

        // 2 periods
        let ap2 = 3;
        let aq2 = 3;

        // Bid price & quantity
        let bp1 = 8;
        let bq1 = 2;

        let bp2 = 6;
        let bq2 = 3;

        // Second period
        let bp2 = 4;
        let bq2 = 1;

        variables! {
            vars:
                0 <= a1 <= 1;
                0 <= a2 <= 1;
                0 <= b1 <= 1;
                0 <= b2 <= 1;
                0 <= b3 <= 1;
        } // variables can also be added dynamically
        let cbc_solver =
            LpSolver(CbcSolver::new().with_temp_solution_file("/tmp/test_lp.sol".to_owned()));
        let social_welfare = 2 * bp1 * bq1 * b1 + bp2 * bq2 * b2
            - ap1 * aq1 * a1 - ap2 * aq2 * a2 * 2;
        let solution = vars.maximise(&social_welfare)
            .using(cbc_solver)
            // first period
            .with(constraint!(bq1 * b1 + bq2 * b2 == aq1 * a1 + aq2 * a2))
            // second period
            .with(constraint!(bq1 * b1 == aq2 * a2))
            .solve().unwrap();
        println!("a1={} a2={} b1={} b2={}", solution.value(a1), solution.value(a2),
                 solution.value(b1), solution.value(b2));
        println!("Social welfare = {}", solution.eval(social_welfare));
        //assert!(false);
    }*/

    #[test]
    fn test_solve_continuous_products() {
        let bid_1_price = 5;
        let bid_1 = (
            product_id(1),
            fixed_load(Product {
                price: bid_1_price,
                quantity: 5,
                start_period: 1,
                end_period: 3,
            }),
        );

        let bid_2_price = 7;
        let bid_2 = (
            product_id(2),
            fixed_load(Product {
                price: bid_2_price,
                quantity: 2,
                start_period: 5,
                end_period: 7,
            }),
        );

        let ask_1 = (
            product_id(3),
            fixed_load(Product {
                price: 4,
                quantity: 5,
                start_period: 1,
                end_period: 3,
            }),
        );
        let ask_2 = (
            product_id(4),
            fixed_load(Product {
                price: 6,
                quantity: 2,
                start_period: 5,
                end_period: 7,
            }),
        );

        let bids = vec![bid_1.clone(), bid_2.clone()];

        let asks = vec![ask_1.clone(), ask_2.clone()];

        let solution = solve(bids, asks, 7).unwrap();
        assert_eq!(
            solution.bids,
            vec![
                (bid_1.0, selected_load_index(0)),
                (bid_2.0, selected_load_index(0))
            ]
        );
        assert_eq!(
            solution.asks,
            vec![
                (ask_1.0, selected_load_index(0)),
                (ask_2.0, selected_load_index(0)),
            ]
        );
        assert_eq!(
            solution.auction_prices,
            vec![0, bid_1_price, bid_1_price, 0, 0, bid_2_price, bid_2_price]
        )
    }

    #[test]
    fn test_solve_overlapping_continuous_products() {
        let bid_1 = (
            product_id(1),
            fixed_load(Product {
                price: 5,
                quantity: 5,
                start_period: 1,
                end_period: 4,
            }),
        );
        let bid_2 = (
            product_id(2),
            fixed_load(Product {
                price: 7,
                quantity: 2,
                start_period: 2,
                end_period: 5,
            }),
        );

        let ask_1 = (
            product_id(3),
            fixed_load(Product {
                price: 4,
                quantity: 7,
                start_period: 1,
                end_period: 5,
            }),
        );

        let bids = vec![bid_1.clone(), bid_2.clone()];

        let asks = vec![ask_1.clone()];

        let solution = solve(bids, asks.clone(), 5).unwrap();
        // No enough bid quantity to match ask_1 at period 1
        assert!(solution.no_solution());

        let bid_3 = (
            product_id(4),
            fixed_load(Product {
                price: 5,
                quantity: 2,
                start_period: 1,
                end_period: 2,
            }),
        );
        let bid_4 = (
            product_id(5),
            fixed_load(Product {
                price: 6,
                quantity: 5,
                start_period: 4,
                end_period: 5,
            }),
        );

        let bids = vec![bid_1.clone(), bid_2.clone(), bid_3.clone(), bid_4.clone()];

        let solution = solve(bids.clone(), asks, 5).unwrap();
        assert_eq!(
            solution.bids,
            vec![
                (bid_1.0, selected_load_index(0)),
                (bid_2.0, selected_load_index(0)),
                (bid_3.0, selected_load_index(0)),
                (bid_4.0, selected_load_index(0)),
            ]
        );
        assert_eq!(solution.asks, vec![(ask_1.0, selected_load_index(0))]);
        assert_eq!(solution.auction_prices, vec![0, 5, 5, 5, 6])
    }

    #[test]
    fn test_solve_flexible_products() {
        let alice_bid = (
            product_id(1),
            vec![
                Product {
                    price: 20,
                    quantity: 5,
                    start_period: 0,
                    end_period: 2,
                },
                Product {
                    price: 16,
                    quantity: 6,
                    start_period: 3,
                    end_period: 5,
                },
            ],
        );
        let alice_ask = (
            product_id(2),
            vec![Product {
                price: 14,
                quantity: 6,
                start_period: 1,
                end_period: 2,
            }],
        );

        let bob_bid = (
            product_id(3),
            vec![Product {
                price: 18,
                quantity: 5,
                start_period: 2,
                end_period: 4,
            }],
        );

        let charlie_ask = (
            product_id(4),
            vec![
                Product {
                    price: 12,
                    quantity: 6,
                    start_period: 0,
                    end_period: 4,
                },
                Product {
                    price: 14,
                    quantity: 5,
                    start_period: 0,
                    end_period: 4,
                },
            ],
        );

        let daniela_bid = (
            product_id(5),
            vec![Product {
                price: 17,
                quantity: 4,
                start_period: 1,
                end_period: 2,
            }],
        );

        let ella_bid = (
            product_id(6),
            vec![Product {
                price: 15,
                quantity: 2,
                start_period: 1,
                end_period: 2,
            }],
        );

        let ella_ask = (
            product_id(7),
            vec![Product {
                price: 11,
                quantity: 3,
                start_period: 3,
                end_period: 5,
            }],
        );

        let bids = vec![
            alice_bid.clone(),
            bob_bid.clone(),
            daniela_bid.clone(),
            ella_bid.clone(),
        ];

        let asks = vec![alice_ask.clone(), charlie_ask.clone(), ella_ask.clone()];

        let solution = solve(bids, asks, 6).unwrap();
        assert_eq!(
            solution.bids,
            vec![
                (alice_bid.0, selected_load_index(0)),
                (bob_bid.0, selected_load_index(0)),
                (daniela_bid.0, selected_load_index(0)),
                (ella_bid.0, selected_load_index(0)),
            ]
        );
        assert_eq!(
            solution.asks,
            vec![
                (alice_ask.0, selected_load_index(0)),
                (charlie_ask.0, selected_load_index(1))
            ]
        );
        assert_eq!(solution.auction_prices, vec![20, 15, 18, 18, 0, 0])
    }

    // Helper function for readability
    fn product_id(id: u32) -> ProductId {
        id
    }

    // Helper function for readability
    fn selected_load_index(id: u32) -> SelectedFlexibleLoad {
        id
    }

    fn fixed_load(product: Product) -> FlexibleProduct {
        vec![product]
    }
}
