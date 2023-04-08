pub mod error;

use std::sync::{Arc, Mutex};

use crate::error::Error;
use good_lp::{
    constraint,
    solvers::lp_solvers::{CbcSolver, LpSolver},
    variables, SolverModel,
};
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

    pub fn submit(&self) -> Result<ProblemId, Error> {
        variables! {
            vars:
                   a <= 1;
              2 <= b <= 4;
        }
        let problem_id = Uuid::new_v4();
        let cbc_solver = LpSolver(
            CbcSolver::new().with_temp_solution_file(format!("/tmp/milp_{}.sol", problem_id)),
        );
        let _ = vars
            .maximise(10 * (a - b / 5) - b)
            .using(cbc_solver) // multiple solvers available
            .with(constraint!(a + 2 <= b))
            .with(constraint!(1 + a >= 4 - b))
            .solve()
            .map_err(|err| Error::Solver(err.to_string()))?;
        let mut active_problems = self.active_problems.lock().unwrap();
        active_problems.push(problem_id);
        Ok(problem_id)
    }
}
