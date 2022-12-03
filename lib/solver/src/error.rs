use hyper::{Body, Response, StatusCode};


#[derive(Clone, Debug)]
pub enum Error {
    SolverError(String),
}

impl Error {
    fn http_body(&self) -> Body {
        match self {
            Self::SolverError(s) => s.clone().into_bytes().into(),
        }
    }

    fn default_err_resp() -> Response<Body> {
        // This should be safe to unwrap
        Response::builder().status(StatusCode::INTERNAL_SERVER_ERROR).body("internal server error".into()).unwrap()
    }
}

impl Into<Response<Body>> for Error {
    fn into(self) -> Response<Body> {
        match Response::builder().status(StatusCode::INTERNAL_SERVER_ERROR).body(self.http_body()) {
            Ok(resp) => resp,
            Err(err) => {
                eprint!("Failed to convert Error into response: {:?}", err);
                Self::default_err_resp()
            },
        }
    }
}