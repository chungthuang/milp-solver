use log::{error, info};
use parachain_client::ParachainClient;
use solver::Solver;
use std::sync::Arc;
use std::{env, time::Duration};

#[tokio::main]
async fn main() {
    env_logger::init();

    let mut handles = vec![];
    let poll_market_secs: u64 = env::var("POLL_MARKET_FREQ_SECS")
        .expect("POLL_MARKET_FREQ_SECS not provided")
        .parse()
        .expect("POLL_MARKET_FREQ_SECS is not u64");
    let rpc_client_url = env::var("RPC_CLIENT_URL").expect("RPC_CLIENT_URL not provided");
    let parachain_client =
        ParachainClient::new(&rpc_client_url).expect("failed to create parachain client");
    let solver = Arc::new(Solver::new());
    let cloned_solver = Arc::clone(&solver);
    handles.push(tokio::spawn(async move {
        tokio::spawn(poll_market_state(
            parachain_client,
            cloned_solver,
            Duration::from_secs(poll_market_secs),
        ));
    }));
    let poll_solution_secs: u64 = env::var("POLL_SOLUTION_FREQ_SECS")
        .expect("POLL_SOLUTION_FREQ_SECS not provided")
        .parse()
        .expect("POLL_SOLUTION_FREQ_SECS is not u64");
    handles.push(tokio::spawn(poll_solution(
        solver,
        Duration::from_secs(poll_solution_secs),
    )));

    futures::future::join_all(handles).await;
}

async fn poll_market_state(
    parachain_client: ParachainClient,
    solver: Arc<Solver>,
    poll_freq: Duration,
) {
    loop {
        info!("Poll market state");
        match parachain_client.get_market_state().await {
            Ok(state) => {
                info!("market state {state:?}");
                let problem_id = solver.submit(state.bids, state.asks);
                info!("submit problem {problem_id:?}");
            }
            Err(err) => {
                error!("failed to get market state {err}");
            }
        };

        tokio::time::sleep(poll_freq).await;
    }
}

async fn poll_solution(solver: Arc<Solver>, poll_freq: Duration) {
    loop {
        info!("Poll market solution");
        tokio::time::sleep(poll_freq).await;
        if let Err(e) = solver.poll() {
            error!("failed to poll market solution {e:?}");
        };
    }
}
