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

const STATUS_ACCEPTED: f64 = 1.;

pub fn solve(
    bids: Vec<(AccountId, Vec<FlexibleProduct>)>,
    asks: Vec<(AccountId, Vec<FlexibleProduct>)>,
    periods: u32,
) -> Result<MarketSolution, Error> {
    let mut vars = ProblemVariables::new();

    let mut total_bid_status: Vec<Variable> = Vec::new();
    // A vector of bid quantities for each period
    let mut total_bid_quantities: Vec<Vec<Expression>> = Vec::with_capacity(periods as usize);
    for _ in 0..periods {
        total_bid_quantities.push(Vec::new());
    }
    let mut total_utilities: Vec<Expression> = Vec::new();
    let mut flexible_load_constraint: Vec<Constraint> = Vec::new();
    for (_account, flexible_bids) in bids.iter() {
        for flexible_bid in flexible_bids.iter() {
            let mut bid_status = vars.add_vector(variable().binary(), flexible_bid.len());
            flexible_load_constraint.push(bid_status.iter().sum::<Expression>().leq(1));
            for (bid, status) in flexible_bid.iter().zip(bid_status.iter()) {
                for period in bid.start_period..bid.end_period {
                    total_bid_quantities[period as usize].push(status.mul(bid.quantity as f64));
                    total_utilities.push(status.mul((bid.quantity * bid.price) as f64));
                }
            }
            total_bid_status.append(&mut bid_status);
        }
    }

    let mut total_ask_status: Vec<Variable> = Vec::new();
    let mut total_ask_quantities: Vec<Vec<Expression>> = Vec::with_capacity(periods as usize);
    for _ in 0..periods {
        total_ask_quantities.push(Vec::new());
    }
    let mut total_costs: Vec<Expression> = Vec::new();
    for (_account, flexible_asks) in asks.iter() {
        for flexible_ask in flexible_asks.iter() {
            let mut ask_status = vars.add_vector(variable().binary(), flexible_ask.len());
            flexible_load_constraint.push(ask_status.iter().sum::<Expression>().leq(1));
            for (ask, status) in flexible_ask.iter().zip(ask_status.iter()) {
                for period in ask.start_period..ask.end_period {
                    total_ask_quantities[period as usize].push(status.mul(ask.quantity as f64));
                    total_costs.push(status.mul((ask.quantity * ask.price) as f64));
                }
            }
            total_ask_status.append(&mut ask_status);
        }
    }

    let social_welfare: Expression = total_utilities.into_iter().sum::<Expression>()
        - total_costs.into_iter().sum::<Expression>();
    debug!("social welfare {:?}", social_welfare);

    let problem_id = Uuid::new_v4();
    debug!("Problem ID {problem_id:?}");
    let cbc_solver =
        LpSolver(CbcSolver::new().with_temp_solution_file(format!("/tmp/milp_{}.sol", problem_id)));

    let mut model = vars.maximise(social_welfare).using(cbc_solver);

    for (period, (total_bids_in_period, total_asks_in_period)) in total_bid_quantities
        .into_iter()
        .zip(total_ask_quantities)
        .enumerate()
    {
        let quantity_match: Expression = total_bids_in_period.into_iter().sum::<Expression>()
            - total_asks_in_period.into_iter().sum::<Expression>();
        debug!("period {period} quantity match constraint {quantity_match:?}");
        model = model.with(quantity_match.eq(0));
    }

    for constraint in flexible_load_constraint.into_iter() {
        model = model.with(constraint);
    }

    let sol = model
        .solve()
        .map_err(|err| Error::Solver(err.to_string()))?;

    evaluate(
        sol,
        bids,
        asks,
        total_bid_status,
        total_ask_status,
        periods as usize,
    )
}

fn evaluate(
    sol: LpSolution,
    bids: Vec<(AccountId, Vec<FlexibleProduct>)>,
    asks: Vec<(AccountId, Vec<FlexibleProduct>)>,
    bid_status: Vec<Variable>,
    ask_status: Vec<Variable>,
    periods: usize,
) -> Result<MarketSolution, Error> {
    let mut auction_prices: Vec<u64> = Vec::with_capacity(periods);
    let mut total_bid_quantities: Vec<u64> = Vec::with_capacity(periods);
    let mut total_ask_quantities: Vec<u64> = Vec::with_capacity(periods);
    for _ in 0..periods {
        auction_prices.push(0);
        total_bid_quantities.push(0);
        total_ask_quantities.push(0);
    }
    let mut bid_status = bid_status.into_iter();
    let mut ask_status = ask_status.into_iter();

    // Iterate the same way as solve creates the vector of bids
    let mut final_bids: Vec<(AccountId, Vec<Option<Product>>)> =
        Vec::with_capacity(bids.len());
    for (account, flexible_bids) in bids.into_iter() {
        let mut account_accepted_bids: Vec<Option<Product>> =
            Vec::with_capacity(flexible_bids.len());
        for flexible_bid in flexible_bids.iter() {
            let mut accepted_bid: Option<Product> = None;
            for bid in flexible_bid.iter() {
                let Some(status) = bid_status.next() else {
                    return Err(Error::InvalidSolution("More bids than bid_status".to_owned()));
                };
                if sol.value(status) == STATUS_ACCEPTED {
                    // Make sure at most only 1 flexible load is accepted
                    if accepted_bid.is_some() {
                        return Err(Error::InvalidSolution(
                            "Multiple flexible load accepted for a bid".to_owned(),
                        ));
                    }
                    accepted_bid = Some(*bid);
                    for period in bid.start_period..bid.end_period {
                        let period = period as usize;
                        if auction_prices[period] == 0 || bid.price < auction_prices[period] {
                            auction_prices[period] = bid.price;
                        }
                        total_bid_quantities[period] += bid.quantity;
                    }
                }
            }
            account_accepted_bids.push(accepted_bid);
        }
        final_bids.push((account, account_accepted_bids));
    }
    if bid_status.next().is_some() {
        return Err(Error::InvalidSolution(
            "More bid_status than bids".to_owned(),
        ));
    }

    let mut final_asks: Vec<(AccountId, Vec<Option<Product>>)> = Vec::with_capacity(asks.len());
    for (account, flexible_asks) in asks.into_iter() {
        let mut account_accepted_asks: Vec<Option<Product>> =
            Vec::with_capacity(flexible_asks.len());
        for flexible_ask in flexible_asks.iter() {
            let mut accepted_ask: Option<Product> = None;
            for ask in flexible_ask.iter() {
                let Some(status) = ask_status.next() else {
                    return Err(Error::InvalidSolution("More asks than ask_status".to_owned()));
                };
                if sol.value(status) == STATUS_ACCEPTED {
                    // Make sure at most only 1 flexible load is accepted
                    if accepted_ask.is_some() {
                        return Err(Error::InvalidSolution(
                            "Multiple flexible load accepted for a ask".to_owned(),
                        ));
                    }
                    accepted_ask = Some(*ask);
                    for period in ask.start_period..ask.end_period {
                        let period = period as usize;
                        if ask.price > auction_prices[period] {
                            // All ask price should be lower than auction price
                            return Err(Error::InvalidSolution(format!(
                                "Ask price {} from account {:?} is higher than auction price {}",
                                ask.price, account, auction_prices[period]
                            )));
                        }
                        total_ask_quantities[period] += ask.quantity;
                    }
                }
            }
            account_accepted_asks.push(accepted_ask);
        }
        final_asks.push((account, account_accepted_asks));
    }
    if ask_status.next().is_some() {
        return Err(Error::InvalidSolution(
            "More ask_status than asks".to_owned(),
        ));
    }

    for (period, (bid_quantity, ask_quantity)) in total_bid_quantities
        .iter()
        .zip(total_ask_quantities)
        .enumerate()
    {
        if *bid_quantity != ask_quantity {
            return Err(Error::InvalidSolution(format!(
                "Total bid quantity {bid_quantity} != total ask quantity {ask_quantity} at period {period}"
            )));
        }
    }

    Ok(MarketSolution {
        bids: final_bids,
        asks: final_asks,
        auction_prices,
    })
}
