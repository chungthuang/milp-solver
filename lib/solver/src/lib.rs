pub mod error;

use std::ops::Mul;

use crate::error::Error;
use good_lp::solvers::lp_solvers::LpSolution;
use good_lp::{
    solvers::lp_solvers::{CbcSolver, LpSolver},
    variable, Constraint, Expression, ProblemVariables, Solution, SolverModel, Variable,
};
use log::debug;
use parachain_client::{AccountId, FlexibleProduct, MarketSolution, Product};
use uuid::Uuid;

const DECISION_ACCEPTED: f64 = 1.;

#[derive(Copy, Clone)]
enum ProductType {
    Bid,
    Ask,
}

pub fn solve(
    bids: Vec<(AccountId, Vec<FlexibleProduct>)>,
    asks: Vec<(AccountId, Vec<FlexibleProduct>)>,
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
            "period {} quantity match constraint {:?}",
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
    products: &[(AccountId, Vec<FlexibleProduct>)],
    periods: usize,
) -> ProblemFormulation {
    let mut decisions: Vec<Variable> = Vec::new();
    let mut flex_load_constraints: Vec<Constraint> = Vec::new();
    let mut quantities: Vec<Vec<Expression>> = Vec::with_capacity(periods as usize);
    for _ in 0..periods {
        quantities.push(Vec::new());
    }
    let mut welfare: Vec<Expression> = Vec::new();
    for (_account, flexible_products) in products.iter() {
        for flexible_product in flexible_products.iter() {
            let mut flex_load_decisions =
                vars.add_vector(variable().binary(), flexible_product.len());
            flex_load_constraints.push(flex_load_decisions.iter().sum::<Expression>().leq(1));
            for (product, decision) in flexible_product.iter().zip(flex_load_decisions.iter()) {
                for period in product.start_period..product.end_period {
                    quantities[period as usize].push(decision.mul(product.quantity as f64));
                    welfare.push(decision.mul((product.quantity * product.price) as f64));
                }
            }
            decisions.append(&mut flex_load_decisions);
        }
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
    bids: Vec<(AccountId, Vec<FlexibleProduct>)>,
    asks: Vec<(AccountId, Vec<FlexibleProduct>)>,
    bid_status: Vec<Variable>,
    ask_status: Vec<Variable>,
    periods: usize,
) -> Result<MarketSolution, Error> {
    let bids_solutions = evaluate_decisions(&sol, bids, bid_status, ProductType::Bid, periods)?;
    let asks_solutions = evaluate_decisions(&sol, asks, ask_status, ProductType::Ask, periods)?;

    let mut auction_prices = Vec::with_capacity(periods);
    for (period, (min_bid_price, max_ask_price)) in bids_solutions.auction_prices.iter().zip(asks_solutions.auction_prices).enumerate() {
        if *min_bid_price < max_ask_price {
            return Err(Error::InvalidSolution(format!("Minimum bid price {} is less than maximum ask price {} at period {}", min_bid_price, max_ask_price, period)))
        }
        auction_prices.push(*min_bid_price);
    }

    for (period, (bid_quantity, ask_quantity)) in bids_solutions.quantities
        .iter()
        .zip(asks_solutions.quantities)
        .enumerate()
    {
        if *bid_quantity != ask_quantity {
            return Err(Error::InvalidSolution(format!(
                "Total bid quantity {} != total ask quantity {} at period {}", bid_quantity, ask_quantity, period
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
    solutions: Vec<(AccountId, Vec<Option<Product>>)>,
    // Auction price by period
    auction_prices: Vec<u64>,
    // Quantity by period
    quantities: Vec<u64>,
}

/// Decide if a product is accepted, and if so, which flexible load
fn evaluate_decisions(
    sol: &LpSolution,
    products: Vec<(AccountId, Vec<FlexibleProduct>)>,
    decisions: Vec<Variable>,
    product_type: ProductType,
    periods: usize,
) -> Result<ProductDecisions, Error> {
    let mut solutions: Vec<(AccountId, Vec<Option<Product>>)> = Vec::with_capacity(products.len());
    let mut auction_prices: Vec<u64> = Vec::with_capacity(periods);
    let mut quantities: Vec<u64> = Vec::with_capacity(periods);
    for _ in 0..periods {
        auction_prices.push(0);
        quantities.push(0);
    }

    let mut decisions = decisions.into_iter();
    // Iterate the same way as solve creates the vector of bids
    for (account, flexible_products) in products.into_iter() {
        let mut account_solutions: Vec<Option<Product>> =
            Vec::with_capacity(flexible_products.len());
        for flexible_product in flexible_products.iter() {
            let mut accepted: Option<Product> = None;
            for product in flexible_product.iter() {
                let Some(decision) = decisions.next() else {
                    return Err(Error::InvalidSolution("More products than decision variables".to_owned()));
                };
                if sol.value(decision) == DECISION_ACCEPTED {
                    // Make sure at most only 1 flexible load is accepted
                    if accepted.is_some() {
                        return Err(Error::InvalidSolution(
                            "Multiple flexible load accepted".to_owned(),
                        ));
                    }
                    accepted = Some(*product);
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
            account_solutions.push(accepted);
        }
        solutions.push((account, account_solutions));
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
