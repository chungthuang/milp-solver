pub mod error;

use error::Error;
use good_lp::{constraint, default_solver, Solution, SolverModel, variables};

pub fn solve() -> Result<String, Error> {
    variables! {
        vars:
               a <= 1;
          2 <= b <= 4;
    } // variables can also be added dynamically
    let solution = vars.maximise(10 * (a - b / 5) - b)
        .using(default_solver) // multiple solvers available
        .with(constraint!(a + 2 <= b))
        .with(constraint!(1 + a >= 4 - b))
        .solve().map_err(|err| Error::SolverError(err.to_string()))?;
    Ok(format!("a={}   b={}", solution.value(a), solution.value(b)))
}