extern crate core;

pub mod error;

use core::fmt;
use std::fmt::Formatter;
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

const DECISION_ACCEPTED: f64 = 1.;
// Used in Big-M constraint for selecting only 1 flexible product
const PERCENTAGE_UPPER_BOUND: f64 = 100.;

#[derive(Copy, Clone)]
enum ProductType {
    Bid,
    Offer,
}

impl fmt::Display for ProductType {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Bid => write!(f, "bid"),
            Self::Offer => write!(f, "offer"),
        }
    }
}

pub fn solve(
    bids: Vec<(ProductId, FlexibleProduct)>,
    offers: Vec<(ProductId, FlexibleProduct)>,
    periods: u32,
    feed_in_tarrif: f64,
    grid_price: f64,
    solution_id: &str,
) -> Result<MarketSolution, Error> {
    // Indicators for flexible load constraint and percentage of a product that's accepted
    let mut vars = ProblemVariables::new();
    let auction_price_vars = vars.add_vector(variable().min(0), periods as usize);

    let bid_formulation = formulate(
        &mut vars,
        &auction_price_vars,
        &bids,
        ProductType::Bid,
        periods as usize,
        grid_price,
    );
    debug!("Utilities {:?}", bid_formulation.welfare);
    let offer_formulation = formulate(
        &mut vars,
        &auction_price_vars,
        &offers,
        ProductType::Offer,
        periods as usize,
        feed_in_tarrif,
    );
    debug!("Costs {:?}", offer_formulation.welfare);

    let social_welfare: Expression = bid_formulation.welfare.sub(offer_formulation.welfare);
    debug!("social welfare {:?}", social_welfare);

    let cbc_solver = LpSolver(
        CbcSolver::new().with_temp_solution_file(format!("/tmp/milp_{}.sol", solution_id)),
    );

    let mut model = vars.maximise(social_welfare).using(cbc_solver);

    for (period, (bids_quantities, asks_quantities)) in bid_formulation
        .quantities_by_period
        .into_iter()
        .zip(offer_formulation.quantities_by_period)
        .enumerate()
    {
        let quantity_match = bids_quantities.sub(asks_quantities);
        debug!(
            "Period {}: quantity match constraint {:?}",
            period, quantity_match
        );
        model = model.with(quantity_match.eq(0));
    }

    /*for (period, (utility, cost)) in bid_formulation
        .welfare_by_period
        .into_iter()
        .zip(offer_formulation.welfare_by_period)
        .enumerate()
    {
        let welfare = utility.sub(cost);
        debug!(
            "Period {}: positive welfare constraint {:?}",
            period, welfare
        );
        model = model.with(welfare.geq(0));
    }*/

    for bid in bid_formulation.constraints.into_iter() {
        debug!(
            "Flexible bid sum to 1 constraint {:?}",
            bid.sum_to_one_constraint
        );
        model = model.with(bid.sum_to_one_constraint);
        debug!("Flexible bid Big-M constraints {:?}", bid.big_m_constraints);
        for big_m in bid.big_m_constraints {
            model = model.with(big_m);
        }
        for auction_price in bid.auction_price_constraints {
            model = model.with(auction_price);
        }
    }

    for offer in offer_formulation.constraints.into_iter() {
        debug!(
            "Flexible offer sum to 1 constraint {:?}",
            offer.sum_to_one_constraint
        );
        model = model.with(offer.sum_to_one_constraint);
        debug!(
            "Flexible offer Big-M constraints {:?}",
            offer.big_m_constraints
        );
        for big_m in offer.big_m_constraints {
            model = model.with(big_m);
        }
        for auction_price in offer.auction_price_constraints {
            model = model.with(auction_price);
        }
    }

    let sol = model
        .solve()
        .map_err(|err| Error::Solver(err.to_string()))?;

    for i in 0..periods {
        debug!(
            "Period {i} auction price {}",
            sol.value(auction_price_vars[i as usize])
        );
    }

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
    quantities_by_period: Vec<Expression>,
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
    // For bids, the auction price has be lower than the bid price if it's accepted
    // For offers, the auction price has to higher than the bid price if it's accepted
    auction_price_constraints: Vec<Constraint>,
}

/// Add decision variables for products to vars, return the decision variables
fn formulate(
    vars: &mut ProblemVariables,
    auction_price_vars: &[Variable],
    products: &[(ProductId, FlexibleProduct)],
    product_type: ProductType,
    periods: usize,
    price_bound: f64,
) -> ProblemFormulation {
    let mut product_vars = Vec::with_capacity(products.len());
    let mut product_constraints = Vec::with_capacity(products.len());
    let mut quantities_by_period: Vec<Expression> = Vec::with_capacity(periods as usize);
    for _ in 0..periods {
        quantities_by_period.push(Expression::default());
    }
    let mut welfare = Expression::default();
    for (id, flexible_product) in products.iter() {
        let (per_product_vars, per_product_constraints) = formulate_per_product(
            vars,
            auction_price_vars,
            &mut quantities_by_period,
            &mut welfare,
            id,
            flexible_product,
            product_type,
            price_bound,
        );
        product_vars.push(per_product_vars);
        product_constraints.push(per_product_constraints);
    }
    ProblemFormulation {
        vars: product_vars,
        constraints: product_constraints,
        quantities_by_period,
        welfare,
    }
}

fn formulate_per_product(
    vars: &mut ProblemVariables,
    auction_price_vars: &[Variable],
    quantities_by_period: &mut Vec<Expression>,
    total_welfare: &mut Expression,
    product_id: &ProductId,
    flex_product: &FlexibleProduct,
    product_type: ProductType,
    price_bound: f64,
) -> (FlexibleProductVars, FlexibleProductConstraints) {
    let mut selected_flex_load: Vec<Variable> = Vec::new();
    let mut accepted_perct: Vec<Variable> = Vec::new();
    let mut big_m_constraints: Vec<Constraint> = Vec::new();
    let mut auction_price_constraints: Vec<Constraint> = Vec::new();

    for product in flex_product.iter() {
        let selected = vars.add(
            variable()
                .binary()
                .name(format!("{}_selected_{}", product_type, product_id)),
        );
        selected_flex_load.push(selected);
        let perct = if product.can_partially_accept {
            vars.add(
                variable()
                    .clamp(0, 1)
                    .name(format!("{}_percentage_{}", product_type, product_id)),
            )
        } else {
            vars.add(
                variable()
                    .binary()
                    .name(format!("{}all_or_none_{}", product_type, product_id)),
            )
        };
        accepted_perct.push(perct);
        // perct = 0, selected_flex_load = 0 -> 0 <= 0
        // But perct = 0, selected_flex_load = 1 -> 0 <= 1
        big_m_constraints.push(perct.into_expression().leq(selected));
        // So we need another constraint
        // If perct = 0, selected_flex_load = 0
        // If perct > 0, selected_flex_load = 1
        big_m_constraints.push(
            selected
                .into_expression()
                .leq(perct.mul(PERCENTAGE_UPPER_BOUND)),
        );

        for period in product.start_period..product.end_period {
            let period = period as usize;
            let ap_constraint = match product_type {
                ProductType::Bid => auction_price_vars[period]
                    .into_expression()
                    .leq(selected.mul(product.price) + (1 - selected) * price_bound),
                ProductType::Offer => auction_price_vars[period]
                    .into_expression()
                    .geq(selected.mul(product.price)),
            };
            auction_price_constraints.push(ap_constraint);
            let quantity = perct.mul(product.quantity);
            quantities_by_period[period].add_assign(quantity.clone());
            total_welfare.add_assign(quantity.mul(product.price));
        }
    }
    let sum_to_one_constraint = selected_flex_load.iter().sum::<Expression>().leq(1);
    (
        FlexibleProductVars {
            selected_flex_load,
            percentage: accepted_perct,
        },
        FlexibleProductConstraints {
            sum_to_one_constraint,
            big_m_constraints,
            auction_price_constraints,
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
    debug!("Bid solutions {:?}", bids_solutions);
    let asks_solutions =
        evaluate_product_decisions(&sol, offers, offer_vars, ProductType::Offer, periods)?;
    debug!("Ask solutions {:?}", asks_solutions);

    let mut auction_prices = Vec::with_capacity(periods);
    for (period, (min_bid_price, max_ask_price)) in bids_solutions
        .auction_prices
        .iter()
        .zip(asks_solutions.auction_prices)
        .enumerate()
    {
        debug!(
            "Period {}: Min bid price {}, max ask price {}",
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
        .into_iter()
        .zip(asks_solutions.quantities)
        .enumerate()
    {
        if bid_quantity == 0. && ask_quantity == 0. {
            continue;
        }
        if bid_quantity / ask_quantity > 1.01 {
            return Err(Error::InvalidSolution(format!(
                "Period {}: Total bid quantity {} / total ask quantity {} > 1.01",
                period, bid_quantity, ask_quantity
            )));
        }
        if bid_quantity / ask_quantity < 0.99 {
            return Err(Error::InvalidSolution(format!(
                "Period {}: Total bid quantity {} / total ask quantity {} < 0.99",
                period, bid_quantity, ask_quantity
            )));
        }
    }

    Ok(MarketSolution {
        bids: bids_solutions.solutions,
        offers: asks_solutions.solutions,
        auction_prices,
    })
}

#[derive(Debug)]
struct ProductDecisions {
    solutions: Vec<AcceptedProduct>,
    // Auction price by period
    auction_prices: Vec<f64>,
    // Quantity by period
    quantities: Vec<f64>,
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
    let mut auction_prices: Vec<f64> = Vec::with_capacity(periods);
    let mut quantities: Vec<f64> = Vec::with_capacity(periods);
    for _ in 0..periods {
        auction_prices.push(0.);
        quantities.push(0.);
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
                if auction_prices[period] == 0. {
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
                quantities[period] += product.quantity * perct
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

    const FEED_IN_TARIFF: f64 = 1.;
    const GRID_PRICE: f64 = 20.;

    // Evaluate solution for single period products
    #[test]
    fn test_solve_single_products() {
        let bid_1_price = 6.;
        let bid_1 = (
            product_id(1),
            fixed_load(Product {
                price: bid_1_price,
                quantity: 5.,
                start_period: 1,
                end_period: 2,
                can_partially_accept: false,
            }),
        );
        let ask_1 = (
            product_id(2),
            fixed_load(Product {
                price: 5.,
                quantity: 5.,
                start_period: 1,
                end_period: 2,
                can_partially_accept: false,
            }),
        );

        let bid_2_price = 8.;
        let bid_2 = (
            product_id(3),
            fixed_load(Product {
                price: bid_2_price,
                quantity: 2.,
                start_period: 3,
                end_period: 4,
                can_partially_accept: false,
            }),
        );
        let ask_2 = (
            product_id(4),
            fixed_load(Product {
                price: 7.,
                quantity: 2.,
                start_period: 3,
                end_period: 4,
                can_partially_accept: false,
            }),
        );

        let bids = vec![bid_1.clone(), bid_2.clone()];

        let asks = vec![ask_1.clone(), ask_2.clone()];

        let solution = solve(
            bids,
            asks,
            5,
            FEED_IN_TARIFF,
            GRID_PRICE,
            "test_solve_single_products",
        )
        .unwrap();
        assert_eq!(
            solution.bids,
            vec![
                accept_product(bid_1.0, 0, 100),
                accept_product(bid_2.0, 0, 100)
            ]
        );
        assert_eq!(
            solution.offers,
            vec![
                accept_product(ask_1.0, 0, 100),
                accept_product(ask_2.0, 0, 100),
            ]
        );
        assert_eq!(
            solution.auction_prices,
            vec![0., bid_1_price, 0., bid_2_price, 0.]
        )
    }

    #[test]
    fn test_solve_single_products_max_social_welfare() {
        let bid_1 = (
            product_id(1),
            fixed_load(Product {
                price: 6.,
                quantity: 5.,
                start_period: 1,
                end_period: 2,
                can_partially_accept: false,
            }),
        );
        let ask_1 = (
            product_id(2),
            fixed_load(Product {
                price: 6.,
                quantity: 5.,
                start_period: 1,
                end_period: 2,
                can_partially_accept: false,
            }),
        );

        let bid_2_price = 7.;
        let bid_2 = (
            product_id(3),
            fixed_load(Product {
                price: bid_2_price,
                quantity: 5.,
                start_period: 1,
                end_period: 2,
                can_partially_accept: false,
            }),
        );
        let ask_2 = (
            product_id(4),
            fixed_load(Product {
                price: 7.,
                quantity: 5.,
                start_period: 1,
                end_period: 2,
                can_partially_accept: false,
            }),
        );

        let bids = vec![bid_1, bid_2.clone()];

        let asks = vec![ask_1.clone(), ask_2];

        // bid_2 will match with ask_1, because the social welfare would be 7 * 5 - 6 * 5 = 5
        // if bid_1 matches with ask_1 and bid_2 matches with ask_2, than the social welfare would be 0
        let solution = solve(
            bids,
            asks,
            3,
            FEED_IN_TARIFF,
            GRID_PRICE,
            "test_solve_single_products_max_social_welfare",
        )
        .unwrap();
        assert_eq!(solution.bids, vec![accept_product(bid_2.0, 0, 100)]);
        assert_eq!(solution.offers, vec![accept_product(ask_1.0, 0, 100)]);
        assert_eq!(solution.auction_prices, vec![0., bid_2_price, 0.])
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
        let bid_1_price = 5.;
        let bid_1 = (
            product_id(1),
            fixed_load(Product {
                price: bid_1_price,
                quantity: 5.,
                start_period: 1,
                end_period: 3,
                can_partially_accept: false,
            }),
        );

        let bid_2_price = 7.;
        let bid_2 = (
            product_id(2),
            fixed_load(Product {
                price: bid_2_price,
                quantity: 2.,
                start_period: 5,
                end_period: 7,
                can_partially_accept: false,
            }),
        );

        let ask_1 = (
            product_id(3),
            fixed_load(Product {
                price: 4.,
                quantity: 5.,
                start_period: 1,
                end_period: 3,
                can_partially_accept: false,
            }),
        );
        let ask_2 = (
            product_id(4),
            fixed_load(Product {
                price: 6.,
                quantity: 2.,
                start_period: 5,
                end_period: 7,
                can_partially_accept: false,
            }),
        );

        let bids = vec![bid_1.clone(), bid_2.clone()];

        let asks = vec![ask_1.clone(), ask_2.clone()];

        let solution = solve(
            bids,
            asks,
            7,
            FEED_IN_TARIFF,
            GRID_PRICE,
            "test_solve_continuous_products",
        )
        .unwrap();
        assert_eq!(
            solution.bids,
            vec![
                accept_product(bid_1.0, 0, 100),
                accept_product(bid_2.0, 0, 100)
            ]
        );
        assert_eq!(
            solution.offers,
            vec![
                accept_product(ask_1.0, 0, 100),
                accept_product(ask_2.0, 0, 100),
            ]
        );
        assert_eq!(
            solution.auction_prices,
            vec![
                0.,
                bid_1_price,
                bid_1_price,
                0.,
                0.,
                bid_2_price,
                bid_2_price
            ]
        )
    }

    #[test]
    fn test_solve_overlapping_continuous_products() {
        let bid_1 = (
            product_id(1),
            fixed_load(Product {
                price: 5.,
                quantity: 5.,
                start_period: 1,
                end_period: 4,
                can_partially_accept: false,
            }),
        );
        let bid_2 = (
            product_id(2),
            fixed_load(Product {
                price: 7.,
                quantity: 2.,
                start_period: 2,
                end_period: 5,
                can_partially_accept: false,
            }),
        );

        let ask_1 = (
            product_id(3),
            fixed_load(Product {
                price: 4.,
                quantity: 7.,
                start_period: 1,
                end_period: 5,
                can_partially_accept: false,
            }),
        );

        let bids = vec![bid_1.clone(), bid_2.clone()];

        let asks = vec![ask_1.clone()];

        let solution = solve(
            bids,
            asks.clone(),
            5,
            FEED_IN_TARIFF,
            GRID_PRICE,
            "test_solve_overlapping_continuous_products",
        )
        .unwrap();
        // No enough bid quantity to match ask_1 at period 1
        assert!(solution.no_solution());

        let bid_3 = (
            product_id(4),
            fixed_load(Product {
                price: 5.,
                quantity: 2.,
                start_period: 1,
                end_period: 2,
                can_partially_accept: false,
            }),
        );
        let bid_4 = (
            product_id(5),
            fixed_load(Product {
                price: 6.,
                quantity: 5.,
                start_period: 4,
                end_period: 5,
                can_partially_accept: false,
            }),
        );

        let bids = vec![bid_1.clone(), bid_2.clone(), bid_3.clone(), bid_4.clone()];

        let solution = solve(
            bids.clone(),
            asks,
            5,
            FEED_IN_TARIFF,
            GRID_PRICE,
            "test_solve_overlapping_continuous_products",
        )
        .unwrap();
        assert_eq!(
            solution.bids,
            vec![
                accept_product(bid_1.0, 0, 100),
                accept_product(bid_2.0, 0, 100),
                accept_product(bid_3.0, 0, 100),
                accept_product(bid_4.0, 0, 100),
            ]
        );
        assert_eq!(solution.offers, vec![accept_product(ask_1.0, 0, 100)]);
        assert_eq!(solution.auction_prices, vec![0., 5., 5., 5., 6.])
    }

    #[test]
    fn test_solve_flexible_products() {
        let alice_bid = (
            product_id(1),
            vec![
                Product {
                    price: 20.,
                    quantity: 5.,
                    start_period: 0,
                    end_period: 2,
                    can_partially_accept: false,
                },
                Product {
                    price: 16.,
                    quantity: 6.,
                    start_period: 3,
                    end_period: 5,
                    can_partially_accept: false,
                },
            ],
        );
        let alice_ask = (
            product_id(2),
            vec![Product {
                price: 14.,
                quantity: 6.,
                start_period: 1,
                end_period: 2,
                can_partially_accept: false,
            }],
        );

        let bob_bid = (
            product_id(3),
            vec![Product {
                price: 18.,
                quantity: 5.,
                start_period: 2,
                end_period: 4,
                can_partially_accept: false,
            }],
        );

        let charlie_ask = (
            product_id(4),
            vec![
                Product {
                    price: 12.,
                    quantity: 6.,
                    start_period: 0,
                    end_period: 4,
                    can_partially_accept: false,
                },
                Product {
                    price: 14.,
                    quantity: 5.,
                    start_period: 0,
                    end_period: 4,
                    can_partially_accept: false,
                },
            ],
        );

        let daniela_bid = (
            product_id(5),
            vec![Product {
                price: 17.,
                quantity: 4.,
                start_period: 1,
                end_period: 2,
                can_partially_accept: false,
            }],
        );

        let ella_bid = (
            product_id(6),
            vec![Product {
                price: 15.,
                quantity: 2.,
                start_period: 1,
                end_period: 2,
                can_partially_accept: false,
            }],
        );

        let ella_ask = (
            product_id(7),
            vec![Product {
                price: 11.,
                quantity: 3.,
                start_period: 3,
                end_period: 5,
                can_partially_accept: false,
            }],
        );

        let bids = vec![
            alice_bid.clone(),
            bob_bid.clone(),
            daniela_bid.clone(),
            ella_bid.clone(),
        ];

        let asks = vec![alice_ask.clone(), charlie_ask.clone(), ella_ask.clone()];

        let solution = solve(
            bids,
            asks,
            6,
            FEED_IN_TARIFF,
            GRID_PRICE,
            "test_solve_flexible_products",
        )
        .unwrap();
        assert_eq!(
            solution.bids,
            vec![
                accept_product(alice_bid.0, 0, 100),
                accept_product(bob_bid.0, 0, 100),
                accept_product(daniela_bid.0, 0, 100),
                accept_product(ella_bid.0, 0, 100),
            ]
        );
        assert_eq!(
            solution.offers,
            vec![
                accept_product(alice_ask.0, 0, 100),
                accept_product(charlie_ask.0, 1, 100)
            ]
        );
        assert_eq!(solution.auction_prices, vec![20., 15., 18., 18., 0., 0.])
    }

    #[test]
    fn test_solve_simple_partial_products() {
        let bid_quantity = 5.;
        let mut bid = (
            product_id(1),
            fixed_load(Product {
                price: 5.,
                quantity: bid_quantity,
                start_period: 0,
                end_period: 3,
                can_partially_accept: false,
            }),
        );

        let ask_quantity = 7.;
        let mut ask = (
            product_id(2),
            fixed_load(Product {
                price: 4.,
                quantity: ask_quantity,
                start_period: 0,
                end_period: 3,
                can_partially_accept: false,
            }),
        );

        let solution = solve(
            vec![bid.clone()],
            vec![ask.clone()],
            3,
            FEED_IN_TARIFF,
            GRID_PRICE,
            "test_solve_simple_partial_products",
        )
        .unwrap();
        assert!(solution.no_solution());

        bid.1[0].can_partially_accept = true;
        ask.1[0].can_partially_accept = true;

        let solution = solve(
            vec![bid.clone()],
            vec![ask.clone()],
            3,
            FEED_IN_TARIFF,
            GRID_PRICE,
            "test_solve_simple_partial_products",
        )
        .unwrap();
        assert_eq!(solution.bids, vec![accept_product(bid.0, 0, 100),]);
        assert_eq!(
            solution.offers,
            vec![accept_product(
                ask.0,
                0,
                (bid_quantity * 100. / ask_quantity).round() as u8
            ),]
        );
        assert_eq!(solution.auction_prices, vec![5., 5., 5.])
    }

    // Data from table 3 of A novel decentralized platform for peer-to-peer energy trading market with
    // blockchain technology paper
    #[test]
    fn test_data_from_paper() {
        let mut offers = Vec::new();
        let account1_p1_a1 = (
            product_id(1),
            fixed_load(Product {
                price: 10.1,
                quantity: 1.5,
                start_period: 0,
                end_period: 1,
                can_partially_accept: true,
            }),
        );
        offers.push(account1_p1_a1);

        let account1_p2_a1 = (
            product_id(2),
            fixed_load(Product {
                price: 4.2,
                quantity: 2.,
                start_period: 1,
                end_period: 2,
                can_partially_accept: true,
            }),
        );
        offers.push(account1_p2_a1);

        let account1_p2_a2 = (
            product_id(3),
            fixed_load(Product {
                price: 6.3,
                quantity: 1.5,
                start_period: 1,
                end_period: 2,
                can_partially_accept: true,
            }),
        );
        offers.push(account1_p2_a2);

        let account1_p3_a1 = (
            product_id(4),
            fixed_load(Product {
                price: 7.9,
                quantity: 1.0,
                start_period: 2,
                end_period: 3,
                can_partially_accept: true,
            }),
        );
        offers.push(account1_p3_a1);

        let account1_p3_a2 = (
            product_id(5),
            fixed_load(Product {
                price: 8.5,
                quantity: 1.5,
                start_period: 2,
                end_period: 3,
                can_partially_accept: true,
            }),
        );
        offers.push(account1_p3_a2);

        let account1_p3_a3 = (
            product_id(6),
            fixed_load(Product {
                price: 9.3,
                quantity: 1.,
                start_period: 2,
                end_period: 3,
                can_partially_accept: true,
            }),
        );
        offers.push(account1_p3_a3);

        let account1_p4_a1 = (
            product_id(7),
            fixed_load(Product {
                price: 5.8,
                quantity: 0.5,
                start_period: 3,
                end_period: 4,
                can_partially_accept: true,
            }),
        );
        offers.push(account1_p4_a1);

        let account1_p4_a2 = (
            product_id(8),
            fixed_load(Product {
                price: 6.8,
                quantity: 0.5,
                start_period: 3,
                end_period: 4,
                can_partially_accept: true,
            }),
        );
        offers.push(account1_p4_a2);

        let account2_p1_a1 = (
            product_id(9),
            fixed_load(Product {
                price: 12.,
                quantity: 1.25,
                start_period: 0,
                end_period: 1,
                can_partially_accept: true,
            }),
        );
        offers.push(account2_p1_a1);

        let account2_p2_a1 = (
            product_id(10),
            fixed_load(Product {
                price: 12.,
                quantity: 1.25,
                start_period: 1,
                end_period: 2,
                can_partially_accept: true,
            }),
        );
        offers.push(account2_p2_a1);

        let account2_p3_a1 = (
            product_id(11),
            fixed_load(Product {
                price: 12.,
                quantity: 1.25,
                start_period: 2,
                end_period: 3,
                can_partially_accept: true,
            }),
        );
        offers.push(account2_p3_a1);

        let account2_p4_a1 = (
            product_id(12),
            fixed_load(Product {
                price: 12.,
                quantity: 1.25,
                start_period: 3,
                end_period: 4,
                can_partially_accept: true,
            }),
        );
        offers.push(account2_p4_a1);

        let account3_a1 = (
            product_id(13),
            fixed_load(Product {
                price: 9.,
                quantity: 2.,
                start_period: 0,
                end_period: 2,
                can_partially_accept: true,
            }),
        );
        offers.push(account3_a1);

        let mut bids = Vec::new();
        let account4_p1_b1 = (
            product_id(1),
            fixed_load(Product {
                price: 7.7,
                quantity: 1.,
                start_period: 0,
                end_period: 1,
                can_partially_accept: true,
            }),
        );
        bids.push(account4_p1_b1);

        let account4_p1_b2 = (
            product_id(2),
            fixed_load(Product {
                price: 6.4,
                quantity: 3.,
                start_period: 0,
                end_period: 1,
                can_partially_accept: true,
            }),
        );
        bids.push(account4_p1_b2);

        let account4_p2_b1 = (
            product_id(3),
            fixed_load(Product {
                price: 8.7,
                quantity: 1.,
                start_period: 1,
                end_period: 2,
                can_partially_accept: true,
            }),
        );
        bids.push(account4_p2_b1);

        let account4_p2_b2 = (
            product_id(4),
            fixed_load(Product {
                price: 7.7,
                quantity: 0.5,
                start_period: 1,
                end_period: 2,
                can_partially_accept: true,
            }),
        );
        bids.push(account4_p2_b2);

        let account4_p2_b3 = (
            product_id(5),
            fixed_load(Product {
                price: 5.3,
                quantity: 1.5,
                start_period: 1,
                end_period: 2,
                can_partially_accept: true,
            }),
        );
        bids.push(account4_p2_b3);

        let account4_p3_b1 = (
            product_id(6),
            fixed_load(Product {
                price: 9.9,
                quantity: 1.25,
                start_period: 2,
                end_period: 3,
                can_partially_accept: true,
            }),
        );
        bids.push(account4_p3_b1);

        let account4_p4_b1 = (
            product_id(7),
            fixed_load(Product {
                price: 7.2,
                quantity: 2.,
                start_period: 3,
                end_period: 4,
                can_partially_accept: true,
            }),
        );
        bids.push(account4_p4_b1);

        let account4_p4_b2 = (
            product_id(8),
            fixed_load(Product {
                price: 6.3,
                quantity: 0.5,
                start_period: 3,
                end_period: 4,
                can_partially_accept: true,
            }),
        );
        bids.push(account4_p4_b2);

        let account5_p1_b1 = (
            product_id(9),
            fixed_load(Product {
                price: 13.5,
                quantity: 1.63,
                start_period: 0,
                end_period: 1,
                can_partially_accept: true,
            }),
        );
        bids.push(account5_p1_b1);

        let account5_p2_b1 = (
            product_id(10),
            fixed_load(Product {
                price: 13.5,
                quantity: 1.63,
                start_period: 1,
                end_period: 2,
                can_partially_accept: true,
            }),
        );
        bids.push(account5_p2_b1);

        let account5_p3_b1 = (
            product_id(11),
            fixed_load(Product {
                price: 13.5,
                quantity: 1.63,
                start_period: 2,
                end_period: 3,
                can_partially_accept: true,
            }),
        );
        bids.push(account5_p3_b1);

        let account5_p4_b1 = (
            product_id(12),
            fixed_load(Product {
                price: 13.5,
                quantity: 1.63,
                start_period: 3,
                end_period: 4,
                can_partially_accept: true,
            }),
        );
        bids.push(account5_p4_b1);

        let account6_b1 = (
            product_id(13),
            fixed_load(Product {
                price: 5.98,
                quantity: 6.,
                start_period: 2,
                end_period: 4,
                can_partially_accept: true,
            }),
        );
        bids.push(account6_b1);

        let solution = solve(
            bids,
            offers,
            4,
            FEED_IN_TARIFF,
            GRID_PRICE,
            "test_data_from_paper",
        )
        .unwrap();
        assert_eq!(solution.bids, vec![
            accept_product(3, 0, 100),
            accept_product(4, 0, 100),
            accept_product(6, 0, 100),
            accept_product(9, 0, 100),
            accept_product(10, 0, 100),
            accept_product(11, 0, 100),
            accept_product(12, 0, 100),
        ]);
        assert_eq!(solution.offers, vec![
            accept_product(1, 0, 100),
            accept_product(2, 0, 100),
            accept_product(3, 0, 75),
            accept_product(4, 0, 100),
            accept_product(5, 0, 100),
            accept_product(6, 0, 38),
            accept_product(7, 0, 100),
            accept_product(8, 0, 100),
            accept_product(9, 0, 10),
            accept_product(12, 0, 50),
        ]);
        assert_eq!(solution.auction_prices, vec![13.5, 7.7, 9.9, 13.5])
    }

    // Helper function for readability
    fn product_id(id: u32) -> ProductId {
        id
    }

    fn fixed_load(product: Product) -> FlexibleProduct {
        vec![product]
    }

    fn accept_product(
        id: ProductId,
        load: SelectedFlexibleLoad,
        percentage: u8,
    ) -> AcceptedProduct {
        AcceptedProduct {
            id,
            load_index: load,
            percentage,
        }
    }
}
