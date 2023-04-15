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
    let parachain_client = ParachainClient::new(&rpc_client_url)
        .await
        .expect("failed to create parachain client");

    loop {
        info!("Poll market state");
        match parachain_client.get_market_state().await {
            Ok(state) => {
                info!("market state {state:?}");
                match solve(state.bids, state.asks) {
                    Ok(solution) => {
                        info!("solution {solution:?}");
                        match parachain_client.submit_solution(solution).await {
                            Ok(hash) => {
                                info!("submit solution in {hash:?}");
                            },
                            Err(err) => {
                                error!("failed to submit solution, err: {err:?}");
                            }
                        }
                    }
                    Err(err) => {
                        error!("failed to get solution, err: {err:?}")
                    }
                };
            }
            Err(err) => {
                error!("failed to get market state, err: {err:?}");
            }
        };

        tokio::time::sleep(poll_freq).await;
    }
}
