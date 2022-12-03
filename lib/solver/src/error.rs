use actix_web::{body::BoxBody, http::StatusCode, HttpResponse, ResponseError};
use std::fmt;

#[derive(Clone, Debug)]
pub enum Error {
    SerializeError(String),
    SolverError(String),
}

impl Error {
    fn http_body(&self) -> BoxBody {
        match self {
            Self::SolverError(s) => BoxBody::new(s.clone()),
            Self::SerializeError(s) => BoxBody::new(s.clone()),
        }
    }
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

impl ResponseError for Error {
    fn status_code(&self) -> StatusCode {
        StatusCode::INTERNAL_SERVER_ERROR
    }

    fn error_response(&self) -> HttpResponse<BoxBody> {
        let body = self.http_body();
        // Create response and set content type
        HttpResponse::InternalServerError().body(body)
    }
}
