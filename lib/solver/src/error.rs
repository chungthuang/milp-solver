use std::fmt;

#[derive(Clone, Debug)]
pub enum Error {
    Serialization(String),
    Solver(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let message = match self {
            Self::Solver(s) => s.clone(),
            Self::Serialization(s) => s.clone(),
        };
        write!(f, "{}", message)
    }
}