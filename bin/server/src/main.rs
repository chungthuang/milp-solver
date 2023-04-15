use log::{error, info};
use parachain_client::ParachainClient;
use solver::solve;
use std::{env, time::Duration};

#[tokio::main]
async fn main() {
    env_logger::init();

    let poll_market_secs: u64 = env::var("POLL_MARKET_FREQ_SECS")
        .expect("POLL_MARKET_FREQ_SECS not provided")
        .parse()
        .expect("POLL_MARKET_FREQ_SECS is not u64");
    let poll_freq = Duration::from_secs(poll_market_secs);
    let rpc_client_url = env::var("RPC_CLIENT_URL").expect("RPC_CLIENT_URL not provided");
    let parachain_client =
        ParachainClient::new(&rpc_client_url).expect("failed to create parachain client");

    loop {
        info!("Poll market state");
        match parachain_client.get_market_state().await {
            Ok(state) => {
                info!("market state {state:?}");
                let solution = solve(state.bids, state.asks);
                info!("solution {solution:?}");
            }
            Err(err) => {
                error!("failed to get market state {err}");
            }
        };

        tokio::time::sleep(poll_freq).await;
    }
}
