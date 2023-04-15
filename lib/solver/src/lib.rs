pub mod error;

use std::ops::Mul;
use std::sync::{Arc, Mutex};

use crate::error::Error;
use good_lp::{
    solvers::lp_solvers::{CbcSolver, LpSolver},
    variable, Expression, ProblemVariables, SolverModel,
};
use parachain_client::Submission;
use uuid::Uuid;

pub struct Solver {
    active_problems: Arc<Mutex<Vec<ProblemId>>>,
}

pub type ProblemId = Uuid;

impl Solver {
    pub fn new() -> Self {
        Solver {
            active_problems: Arc::new(Mutex::new(Vec::new())),
        }
    }

    pub fn submit(&self, bids: Vec<Submission>, asks: Vec<Submission>) -> Result<ProblemId, Error> {
        let mut vars = ProblemVariables::new();

        let bid_accepted = vars.add_vector(variable().binary(), bids.len());
        let ask_accepted = vars.add_vector(variable().binary(), asks.len());

        let bid_quantities = bids.iter().map(|(_, quantity, _)| *quantity as f64);
        let ask_quantities = asks.iter().map(|(_, quantity, _)| *quantity as f64);

        let bid_quantities: Vec<Expression> = bid_quantities.zip(bid_accepted.clone()).map(|(quantity, accepted)| accepted.mul(quantity)).collect();
        let ask_quantities: Vec<Expression> = ask_quantities.zip(ask_accepted.clone()).map(|(quantity, accepted)| accepted.mul(quantity)).collect();

        let bid_utilities = bids.iter().map(|(_, q, p)| (q * p) as f64);
        let ask_costs = asks.iter().map(|(_, q, p)| (q * p) as f64);


        let total_utility: Vec<Expression> = bid_utilities
            .zip(bid_accepted)
            .map(|(utility, accepted)| accepted.mul(utility))
            .collect();
        let total_costs: Vec<Expression> = ask_costs
            .zip(ask_accepted)
            .map(|(cost, accepted)| accepted.mul(cost))
            .collect();

        let social_welfare: Expression =
            total_utility.into_iter().sum::<Expression>() - total_costs.into_iter().sum::<Expression>();

        let quantity_match: Expression = bid_quantities.into_iter().sum::<Expression>() - ask_quantities.into_iter().sum::<Expression>();

        let problem_id = Uuid::new_v4();
        let cbc_solver = LpSolver(
            CbcSolver::new().with_temp_solution_file(format!("/tmp/milp_{}.sol", problem_id)),
        );
        let _ = vars
            .maximise(social_welfare)
            .using(cbc_solver)
            .with(quantity_match.eq(0))
            .solve()
            .map_err(|err| Error::Solver(err.to_string()))?;
        self.active_problems.lock().unwrap().push(problem_id);
        Ok(problem_id)
    }

    pub fn poll(&self) -> Result<(), Error> {
        for problem in self.active_problems.lock().unwrap().iter() {
            println!("problem {problem}");
        }
        Ok(())
    }
}
