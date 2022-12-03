use actix_web::{web, App, HttpRequest, HttpServer, Responder};
use solver::Solver;
use std::sync::Arc;

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    HttpServer::new(|| {
        let app_state = AppState {
            solver: Arc::new(Solver::new()),
        };

        App::new()
            .configure(configure_route)
            .app_data(web::Data::new(app_state))
    })
    .bind(("127.0.0.1", 8080))?
    .run()
    .await
}

// this function could be located in a different module
fn configure_route(cfg: &mut web::ServiceConfig) {
    cfg.service(web::resource("/submit").route(web::get().to(submit)));
}

#[derive(Clone)]
struct AppState {
    solver: Arc<Solver>,
}

async fn submit(_req: HttpRequest, state: web::Data<AppState>) -> impl Responder {
    state.solver.submit()
}
