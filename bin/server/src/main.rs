use log::{error, info};
use parachain_client::ParachainClient;
use solver::solve;
use std::{env, time::Duration};
use uuid::Uuid;

#[tokio::main]
async fn main() {
    env_logger::init();

    let poll_market_secs: u64 = env::var("POLL_MARKET_FREQ_SECS")
        .expect("POLL_MARKET_FREQ_SECS not provided")
        .parse()
        .expect("POLL_MARKET_FREQ_SECS is not u64");
    let poll_freq = Duration::from_secs(poll_market_secs);
    let rpc_client_addr = env::var("RPC_ADDRESS").expect("RPC_ADDRESS not provided");
    let parachain_client = ParachainClient::new(&rpc_client_addr)
        .await
        .expect("failed to create parachain client");

    loop {
        info!("Poll market state");
        match parachain_client.get_market_state().await {
            Ok(state) => {
                info!("market state {state:?}");

                let solution_id = Uuid::new_v4();
                info!("Solution ID {:?}", solution_id);
                match solve(
                    state.bids,
                    state.asks,
                    state.periods,
                    state.feed_in_tariff as f64,
                    state.grid_price as f64,
                    &solution_id.to_string(),
                ) {
                    Ok(solution) => {
                        info!("solution {solution:?}");
                        match parachain_client.submit_solution(solution).await {
                            Ok(hash) => {
                                info!("submit solution in {hash:?}");
                            }
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
