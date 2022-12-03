use crate::error::Error;

use serde::{Deserialize, Serialize};
use hyper::{header, Body, Response};
use uuid::Uuid;

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Submission {
    pub id: Uuid,
}

impl Submission {
    fn http_body(&self) -> Result<Body, Error> {
        serde_json::to_string(self).map(|body| body.into()).map_err(|err| Error::SerializeError(err.to_string()))
    }
}

impl Into<Response<Body>> for Submission {
    fn into(self) -> Response<Body> {
        let body = match self.http_body() {
            Ok(body) => body,
            Err(err) => {
                eprint!("Failed to create submission body: {:?}", err);
                return err.into()
            },
        };
        match Response::builder().header(header::CONTENT_TYPE, "application/json").body(body) {
            Ok(resp) => resp,
            Err(err) => {
                eprint!("Failed to convert Submission into response: {:?}", err);
                Error::default_err_resp()
            },
        }
    }
}