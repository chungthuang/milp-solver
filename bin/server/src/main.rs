use log::info;
use parachain_client::ParachainClient;
use solver::Solver;
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
    handles.push(tokio::spawn(async move {
        tokio::spawn(poll_market_state(
            rpc_client_url,
            Duration::from_secs(poll_market_secs),
        ));
    }));
    let poll_solution_secs: u64 = env::var("POLL_SOLUTION_FREQ_SECS")
        .expect("POLL_SOLUTION_FREQ_SECS not provided")
        .parse()
        .expect("POLL_SOLUTION_FREQ_SECS is not u64");
    handles.push(tokio::spawn(poll_solution(Duration::from_secs(
        poll_solution_secs,
    ))));

    futures::future::join_all(handles).await;
}

async fn poll_market_state(rpc_client_url: String, poll_freq: Duration) {
    loop {
        info!("Poll market state");
        let parachain_client =
            ParachainClient::new(&rpc_client_url).await.expect("failed to create parachain client");
        let state = parachain_client.get_market_state().await;

        tokio::time::sleep(poll_freq).await;
        info!("market state {:?}", state);
    }
}

async fn poll_solution(poll_freq: Duration) {
    let solver = Solver::new();
    loop {
        info!("Poll market solution");
        tokio::time::sleep(poll_freq).await;
        let problem_id = solver.submit();
        info!("Problem ID {:?}", problem_id);
    }
}
