pub mod error;

use std::ops::Mul;

use crate::error::Error;
use good_lp::solvers::lp_solvers::LpSolution;
use good_lp::{
    solvers::lp_solvers::{CbcSolver, LpSolver},
    variable, Expression, ProblemVariables, Solution, SolverModel, Variable,
};
use parachain_client::{AccountId, MarketSolution, Submission};
use uuid::Uuid;

const STATUS_ACCEPTED: f64 = 1.;

pub fn solve(bids: Vec<Submission>, asks: Vec<Submission>) -> Result<MarketSolution, Error> {
    let mut vars = ProblemVariables::new();

    let bid_status = vars.add_vector(variable().binary(), bids.len());
    let ask_status = vars.add_vector(variable().binary(), asks.len());

    let bid_quantities = bids.iter().map(|(_, quantity, _)| *quantity as f64);
    let ask_quantities = asks.iter().map(|(_, quantity, _)| *quantity as f64);

    let bid_quantities: Vec<Expression> = bid_quantities
        .zip(bid_status.clone())
        .map(|(quantity, accepted)| accepted.mul(quantity))
        .collect();
    let ask_quantities: Vec<Expression> = ask_quantities
        .zip(ask_status.clone())
        .map(|(quantity, accepted)| accepted.mul(quantity))
        .collect();

    let bid_utilities = bids.iter().map(|(_, q, p)| (q * p) as f64);
    let ask_costs = asks.iter().map(|(_, q, p)| (q * p) as f64);

    let total_utility: Vec<Expression> = bid_utilities
        .zip(bid_status.clone())
        .map(|(utility, accepted)| accepted.mul(utility))
        .collect();
    let total_costs: Vec<Expression> = ask_costs
        .zip(ask_status.clone())
        .map(|(cost, accepted)| accepted.mul(cost))
        .collect();

    let social_welfare: Expression =
        total_utility.into_iter().sum::<Expression>() - total_costs.into_iter().sum::<Expression>();

    let quantity_match: Expression = bid_quantities.into_iter().sum::<Expression>()
        - ask_quantities.into_iter().sum::<Expression>();

    let problem_id = Uuid::new_v4();
    println!("Problem ID {problem_id:?}");
    let cbc_solver =
        LpSolver(CbcSolver::new().with_temp_solution_file(format!("/tmp/milp_{}.sol", problem_id)));
    let sol = vars
        .maximise(social_welfare)
        .using(cbc_solver)
        .with(quantity_match.eq(0))
        .solve()
        .map_err(|err| Error::Solver(err.to_string()))?;

    evaluate(sol, bids, asks, bid_status, ask_status)
}

fn evaluate(
    sol: LpSolution,
    bids: Vec<Submission>,
    asks: Vec<Submission>,
    bid_status: Vec<Variable>,
    ask_status: Vec<Variable>,
) -> Result<MarketSolution, Error> {
    let mut auction_price = u64::MAX;

    let mut accepted_bids: Vec<AccountId> = Vec::new();
    let mut total_bid_quantity = 0;
    for ((account, quantity, price), _) in bids
        .into_iter()
        .zip(bid_status)
        .filter(|(_, accepted)| sol.value(*accepted) == STATUS_ACCEPTED)
    {
        if price < auction_price {
            auction_price = price;
        }
        total_bid_quantity += quantity;
        accepted_bids.push(account);
    }

    let mut accepted_asks: Vec<AccountId> = Vec::new();
    let mut total_ask_quantity = 0;
    for ((account, quantity, price), _) in asks
        .into_iter()
        .zip(ask_status)
        .filter(|(_, accepted)| sol.value(*accepted) == STATUS_ACCEPTED)
    {
        if price > auction_price {
            // All ask price should be lower than auction price
            return Err(Error::InvalidSolution(format!("Ask price {price} from account {account:?} is higher than auction price {auction_price}")));
        }
        total_ask_quantity += quantity;
        accepted_asks.push(account);
    }

    if total_bid_quantity != total_ask_quantity {
        return Err(Error::InvalidSolution(format!(
            "Total bid quantity {total_bid_quantity} != total ask quantity {total_ask_quantity}"
        )));
    }

    Ok(MarketSolution {
        accepted_bids,
        accepted_asks,
        auction_price,
    })
}
