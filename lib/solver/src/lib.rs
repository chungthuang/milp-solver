pub mod error;

use std::ops::Mul;

use crate::error::Error;
use good_lp::solvers::lp_solvers::LpSolution;
use good_lp::{
    solvers::lp_solvers::{CbcSolver, LpSolver},
    variable, Constraint, Expression, ProblemVariables, Solution, SolverModel, Variable,
};
use log::debug;
use parachain_client::{FlexibleProduct, MarketSolution, ProductId, SelectedFlexibleLoad};
use uuid::Uuid;

const DECISION_ACCEPTED: f64 = 1.;

#[derive(Copy, Clone)]
enum ProductType {
    Bid,
    Ask,
}

pub fn solve(
    bids: Vec<(ProductId, FlexibleProduct)>,
    asks: Vec<(ProductId, FlexibleProduct)>,
    periods: u32,
) -> Result<MarketSolution, Error> {
    let mut vars = ProblemVariables::new();

    let bid_formulation = formulate(&mut vars, &bids, periods as usize);
    let ask_formulation = formulate(&mut vars, &asks, periods as usize);

    let social_welfare: Expression = bid_formulation.welfare.into_iter().sum::<Expression>()
        - ask_formulation.welfare.into_iter().sum::<Expression>();
    debug!("social welfare {:?}", social_welfare);

    let problem_id = Uuid::new_v4();
    debug!("Problem ID {:?}", problem_id);
    let cbc_solver =
        LpSolver(CbcSolver::new().with_temp_solution_file(format!("/tmp/milp_{}.sol", problem_id)));

    let mut model = vars.maximise(social_welfare).using(cbc_solver);

    for (period, (bids_quantities, asks_quantities)) in bid_formulation
        .quantities
        .into_iter()
        .zip(ask_formulation.quantities)
        .enumerate()
    {
        let quantity_match: Expression = bids_quantities.into_iter().sum::<Expression>()
            - asks_quantities.into_iter().sum::<Expression>();
        debug!(
            "Period {}: quantity match constraint {:?}",
            period, quantity_match
        );
        model = model.with(quantity_match.eq(0));
    }

    for constraint in bid_formulation.flex_load_constraints.into_iter() {
        model = model.with(constraint);
    }
    for constraint in ask_formulation.flex_load_constraints.into_iter() {
        model = model.with(constraint);
    }

    let sol = model
        .solve()
        .map_err(|err| Error::Solver(err.to_string()))?;

    evaluate(
        sol,
        bids,
        asks,
        bid_formulation.decisions,
        ask_formulation.decisions,
        periods as usize,
    )
}

struct ProblemFormulation {
    decisions: Vec<Variable>,
    flex_load_constraints: Vec<Constraint>,
    // A vector of bid/ask quantities for each period
    quantities: Vec<Vec<Expression>>,
    // Utility/cost
    welfare: Vec<Expression>,
}

/// Add decision variables for products to vars, return the decision variables
fn formulate(
    vars: &mut ProblemVariables,
    products: &[(ProductId, FlexibleProduct)],
    periods: usize,
) -> ProblemFormulation {
    let mut decisions: Vec<Variable> = Vec::new();
    let mut flex_load_constraints: Vec<Constraint> = Vec::new();
    let mut quantities: Vec<Vec<Expression>> = Vec::with_capacity(periods as usize);
    for _ in 0..periods {
        quantities.push(Vec::new());
    }
    let mut welfare: Vec<Expression> = Vec::new();
    for (_id, flexible_product) in products.iter() {
        let mut flex_load_decisions = vars.add_vector(variable().binary(), flexible_product.len());
        flex_load_constraints.push(flex_load_decisions.iter().sum::<Expression>().leq(1));
        for (product, decision) in flexible_product.iter().zip(flex_load_decisions.iter()) {
            for period in product.start_period..product.end_period {
                quantities[period as usize].push(decision.mul(product.quantity as f64));
                welfare.push(decision.mul((product.quantity * product.price) as f64));
            }
        }
        decisions.append(&mut flex_load_decisions);
    }
    ProblemFormulation {
        decisions,
        flex_load_constraints,
        quantities,
        welfare,
    }
}

fn evaluate(
    sol: LpSolution,
    bids: Vec<(ProductId, FlexibleProduct)>,
    asks: Vec<(ProductId, FlexibleProduct)>,
    bid_status: Vec<Variable>,
    ask_status: Vec<Variable>,
    periods: usize,
) -> Result<MarketSolution, Error> {
    let bids_solutions =
        evaluate_product_decisions(&sol, bids, bid_status, ProductType::Bid, periods)?;
    let asks_solutions =
        evaluate_product_decisions(&sol, asks, ask_status, ProductType::Ask, periods)?;

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
    solutions: Vec<(ProductId, SelectedFlexibleLoad)>,
    // Auction price by period
    auction_prices: Vec<u64>,
    // Quantity by period
    quantities: Vec<u64>,
}

/// Decide if a product is accepted, and if so, which flexible load
fn evaluate_product_decisions(
    sol: &LpSolution,
    products: Vec<(ProductId, FlexibleProduct)>,
    decisions: Vec<Variable>,
    product_type: ProductType,
    periods: usize,
) -> Result<ProductDecisions, Error> {
    let mut solutions: Vec<(ProductId, SelectedFlexibleLoad)> = Vec::with_capacity(products.len());
    let mut auction_prices: Vec<u64> = Vec::with_capacity(periods);
    let mut quantities: Vec<u64> = Vec::with_capacity(periods);
    for _ in 0..periods {
        auction_prices.push(0);
        quantities.push(0);
    }

    let mut decisions = decisions.into_iter();
    // Iterate the same way as solve creates the vector of bids
    for (id, flexible_product) in products.into_iter() {
        let mut accepted_schedule: Option<SelectedFlexibleLoad> = None;
        for (schedule_index, product) in flexible_product.iter().enumerate() {
            let Some(decision) = decisions.next() else {
                return Err(Error::InvalidSolution("More products than decision variables".to_owned()));
            };
            if sol.value(decision) == DECISION_ACCEPTED {
                // Make sure at most only 1 flexible load is accepted
                if accepted_schedule.is_some() {
                    return Err(Error::InvalidSolution(
                        "Multiple flexible load accepted".to_owned(),
                    ));
                }
                accepted_schedule = Some(schedule_index as SelectedFlexibleLoad);
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
                            ProductType::Ask => {
                                if product.price > auction_prices[period] {
                                    auction_prices[period] = product.price;
                                }
                            }
                        };
                    }
                    quantities[period] += product.quantity;
                }
            }
        }
        if let Some(schedule) = accepted_schedule {
            solutions.push((id, schedule));
        }
    }
    if decisions.next().is_some() {
        return Err(Error::InvalidSolution(
            "More decision variables thatn products".to_owned(),
        ));
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
        let bid_1 = (
            product_id(1),
            vec![
                Product {
                    price: 5,
                    quantity: 5,
                    start_period: 1,
                    end_period: 3,
                },
                Product {
                    price: 3,
                    quantity: 3,
                    start_period: 2,
                    end_period: 4,
                },
            ],
        );
        let bid_2 = (
            product_id(2),
            vec![
                Product {
                    price: 8,
                    quantity: 2,
                    start_period: 5,
                    end_period: 7,
                },
                Product {
                    price: 7,
                    quantity: 2,
                    start_period: 2,
                    end_period: 4,
                },
            ],
        );

        let ask_1 = (
            product_id(3),
            vec![Product {
                price: 6,
                quantity: 2,
                start_period: 2,
                end_period: 4,
            }],
        );
        let ask_2 = (
            product_id(4),
            vec![Product {
                price: 2,
                quantity: 3,
                start_period: 1,
                end_period: 5,
            }],
        );

        let bids = vec![bid_1, bid_2.clone()];

        let asks = vec![ask_1.clone(), ask_2];

        let solution = solve(bids, asks, 7).unwrap();
        assert_eq!(solution.bids, vec![(bid_2.0, selected_load_index(1))]);
        assert_eq!(solution.asks, vec![(ask_1.0, selected_load_index(0))]);
        assert_eq!(solution.auction_prices, vec![0, 0, 7, 7, 0, 0, 0])
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
