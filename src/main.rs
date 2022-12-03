use std::convert::Infallible;
use std::net::SocketAddr;
use good_lp::{constraint, default_solver, Solution, SolverModel, variables};
use hyper::{Body, Request, Response, Server, StatusCode};
use hyper::service::{make_service_fn, service_fn};

async fn hello_world(_req: Request<Body>) -> Result<Response<Body>, Infallible> {
    let solution = match solve() {
        Ok(sol) => sol,
        Err(err) => {
            eprint!("solver error: {:?}", err);
            return Ok(err.into())
        }
    };
    let body = solution.into_bytes();
    Ok(Response::new(body.into()))
}

#[derive(Clone, Debug)]
enum Error {
    SolverError(String),
}

impl Error {
    fn http_body(&self) -> Body {
        match self {
            Self::SolverError(s) => s.clone().into_bytes().into(),
        }
    }

    fn default_err_resp() -> Response<Body> {
        Response::builder().status(StatusCode::INTERNAL_SERVER_ERROR).body("internal server error".into()).unwrap()
    }
}

impl Into<Response<Body>> for Error {
    fn into(self) -> Response<Body> {
        match Response::builder().status(StatusCode::INTERNAL_SERVER_ERROR).body(self.http_body()) {
            Ok(resp) => resp,
            // This should be safe to unwrap
            Err(err) => {
                eprint!("Failed to convert Error into response: {:?}", err);
                Self::default_err_resp()
            },
        }
    }
}

#[tokio::main]
async fn main() {
    let addr = SocketAddr::from(([127, 0, 0, 1], 8000));

    // A `Service` is needed for every connection, so this
    // creates one from our `hello_world` function.
    let make_svc = make_service_fn(|_conn| async {
        // service_fn converts our function into a `Service`
        Ok::<_, Infallible>(service_fn(hello_world))
    });

    let server = Server::bind(&addr).serve(make_svc);

    // Run this server for... forever!
    if let Err(e) = server.await {
        eprintln!("server error: {}", e);
    }
}

fn solve() -> Result<String, Error> {
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