use crate::error::Error;

use actix_web::{
    body::BoxBody, http::header::ContentType, HttpRequest, HttpResponse, Responder, ResponseError,
};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Submission {
    pub id: Uuid,
}

impl Submission {
    fn http_body(&self) -> Result<BoxBody, Error> {
        serde_json::to_string(self)
            .map(|s| BoxBody::new(s))
            .map_err(|err| Error::SerializeError(err.to_string()))
    }
}

impl Responder for Submission {
    type Body = BoxBody;

    fn respond_to(self, _req: &HttpRequest) -> HttpResponse<Self::Body> {
        let body = match self.http_body() {
            Ok(body) => body,
            Err(err) => {
                eprint!("Failed to create submission body: {:?}", err);
                return err.error_response();
            }
        };

        // Create response and set content type
        HttpResponse::Ok()
            .content_type(ContentType::json())
            .body(body)
    }
}
