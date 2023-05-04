pub mod error;

use std::ops::Mul;

use crate::error::Error;
use good_lp::solvers::lp_solvers::LpSolution;
use good_lp::{
    solvers::lp_solvers::{CbcSolver, LpSolver},
    variable, Expression, ProblemVariables, Solution, SolverModel, Variable,
};
use log::debug;
use parachain_client::{AccountId, MarketSolution, Product, ProductAccepted};
use uuid::Uuid;

const STATUS_ACCEPTED: f64 = 1.;

pub fn solve(bids: Vec<(AccountId, Vec<Product>)>, asks: Vec<(AccountId, Vec<Product>)>, periods: u32) -> Result<MarketSolution, Error> {
    let mut vars = ProblemVariables::new();

    let mut total_bid_status: Vec<Variable> = Vec::new();
    // A vector of bid quantities for each period
    let mut total_bid_quantities: Vec<Vec<Expression>> = Vec::with_capacity(periods as usize);
    for _ in 0..periods {
        total_bid_quantities.push(Vec::new());
    }
    let mut total_utilities: Vec<Expression> = Vec::new();
    for (_account, account_bids) in bids.iter() {
        for bid in account_bids.iter() {
            let bid_status = vars.add(variable().binary());
            total_bid_status.push(bid_status);
            for period in bid.start_period..bid.end_period {
                total_bid_quantities[period as usize].push(bid_status.mul(bid.quantity as f64));
                total_utilities.push(bid_status.mul((bid.quantity * bid.price) as f64));
            }
        }
    }

    let mut total_ask_status: Vec<Variable> = Vec::new();
    let mut total_ask_quantities: Vec<Vec<Expression>> = Vec::with_capacity(periods as usize);
    for _ in 0..periods {
        total_ask_quantities.push(Vec::new());
    }
    let mut total_costs: Vec<Expression> = Vec::new();
    for (_account, account_asks) in asks.iter() {
        for ask in account_asks.iter() {
            let ask_status = vars.add(variable().binary());
            total_ask_status.push(ask_status);
            for period in ask.start_period..ask.end_period {
                total_ask_quantities[period as usize].push(ask_status.mul(ask.quantity as f64));
                total_costs.push(ask_status.mul((ask.quantity * ask.price) as f64));
            }
        }
    }

    let social_welfare: Expression =
        total_utilities.into_iter().sum::<Expression>() - total_costs.into_iter().sum::<Expression>();
    debug!("social welfare {:?}", social_welfare);

    let problem_id = Uuid::new_v4();
    debug!("Problem ID {problem_id:?}");
    let cbc_solver =
        LpSolver(CbcSolver::new().with_temp_solution_file(format!("/tmp/milp_{}.sol", problem_id)));

    let mut model = vars
        .maximise(social_welfare)
        .using(cbc_solver);

    for (period, (total_bids_in_period, total_asks_in_period)) in total_bid_quantities.into_iter().zip(total_ask_quantities).enumerate() {
        let quantity_match: Expression = total_bids_in_period.into_iter().sum::<Expression>()
            - total_asks_in_period.into_iter().sum::<Expression>();
        debug!("period {period} quantity match constraint {quantity_match:?}");
        model = model.with(quantity_match.eq(0));
    }

    let sol = model.solve().map_err(|err| Error::Solver(err.to_string()))?;

    evaluate(sol, bids, asks, total_bid_status, total_ask_status, periods as usize)
}

fn evaluate(
    sol: LpSolution,
    bids: Vec<(AccountId, Vec<Product>)>,
    asks: Vec<(AccountId, Vec<Product>)>,
    mut bid_status: Vec<Variable>,
    mut ask_status: Vec<Variable>,
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

    // Iterate the same way as solve creates the vector of bids
    let mut solved_bids: Vec<(AccountId, Vec<ProductAccepted>)> = Vec::with_capacity(bids.len());
    for (account, account_bids) in bids.into_iter() {
        let mut account_solved_bids: Vec<ProductAccepted> = Vec::with_capacity(account_bids.len());
        for bid in account_bids.iter() {
            let Some(status) = bid_status.pop() else {
                return Err(Error::InvalidSolution("More bids than bid_status".to_owned()));
            };
            let product_accepted = sol.value(status) == STATUS_ACCEPTED;
            account_solved_bids.push(product_accepted);
            if product_accepted {
                for period in bid.start_period..bid.end_period {
                    let period = period as usize;
                    if auction_prices[period] == 0  || bid.price < auction_prices[period] {
                        auction_prices[period] = bid.price;
                    }
                    total_bid_quantities[period] += bid.quantity;
                }
            }
        }
        solved_bids.push((account, account_solved_bids));
    }
    if let Some(_) = bid_status.pop() {
        return Err(Error::InvalidSolution("More bid_status than bids".to_owned()))
    }

    let mut solved_asks: Vec<(AccountId, Vec<ProductAccepted>)> = Vec::with_capacity(asks.len());
    for (account, account_asks) in asks.into_iter() {
        let mut account_solved_asks: Vec<ProductAccepted> = Vec::with_capacity(account_asks.len());
        for ask in account_asks.iter() {
            let Some(status) = ask_status.pop() else {
                return Err(Error::InvalidSolution("More asks than ask_status".to_owned()));
            };
            let product_accepted = sol.value(status) == STATUS_ACCEPTED;
            account_solved_asks.push(product_accepted);
            if product_accepted {
                for period in ask.start_period..ask.end_period {
                    let period = period as usize;
                    if ask.price > auction_prices[period] {
                        // All ask price should be lower than auction price
                        return Err(Error::InvalidSolution(format!("Ask price {} from account {:?} is higher than auction price {}", ask.price, account, auction_prices[period])));
                    }
                    total_ask_quantities[period] += ask.quantity;
                }
            }
        }
        solved_asks.push((account, account_solved_asks));
    }
    if let Some(_) = ask_status.pop() {
        return Err(Error::InvalidSolution("More ask_status than asks".to_owned()))
    }

    for (period, (bid_quantity, ask_quantity)) in total_bid_quantities.iter().zip(total_ask_quantities).enumerate() {
        if *bid_quantity != ask_quantity {
            return Err(Error::InvalidSolution(format!(
                "Total bid quantity {bid_quantity} != total ask quantity {ask_quantity} at period {period}"
            )));
        }
    }

    Ok(MarketSolution {
        bids: solved_bids,
        asks: solved_asks,
        auction_prices,
    })
}
