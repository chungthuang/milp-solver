pub mod error;
pub mod response;

use crate::error::Error;
use crate::response::Submission;
use good_lp::{
    constraint,
    solvers::lp_solvers::{CbcSolver, LpSolver},
    variables, SolverModel,
};
use uuid::Uuid;

pub struct Solver {}

impl Solver {
    pub fn new() -> Self {
        Solver {}
    }

    pub fn submit(&self) -> Result<Submission, Error> {
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
            .map_err(|err| Error::SolverError(err.to_string()))?;
        Ok(Submission { id: problem_id })
    }
}
