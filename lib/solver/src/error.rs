use actix_web::{body::BoxBody, http::StatusCode, HttpResponse, ResponseError};
use std::fmt;

#[derive(Clone, Debug)]
pub enum Error {
    Serialization(String),
    // Encounter error running the solver
    Solver(String),
    // Solver returned an invalid solution
    InvalidSolution(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let message = match self {
            Self::Solver(s) => s.clone(),
            Self::Serialization(s) => s.clone(),
            Self::InvalidSolution(s) => s.clone(),
        };
        write!(f, "{}", message)
    }
}
