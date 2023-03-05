use std::fmt;

#[derive(Clone, Debug)]
pub enum Error {
    SerializeError(String),
    SolverError(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let message = match self {
            Self::SolverError(s) => s.clone(),
            Self::SerializeError(s) => s.clone(),
        };
        write!(f, "{}", message)
    }
}