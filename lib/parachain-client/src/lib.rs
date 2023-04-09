use anyhow::{anyhow, Result};
use jsonrpsee::rpc_params;
use jsonrpsee::{
    core::client::ClientT,
    ws_client::{WsClient, WsClientBuilder},
};
use serde::Deserialize;

const CLEARING_STAGE: Stage = 1;

pub struct ParachainClient {
    rpc_client: WsClient,
}

type Stage = u64;

#[derive(Debug, Deserialize)]
pub struct MarketState {
    pub bids: Vec<Submission>,
    pub asks: Vec<Submission>,
    pub stage: Stage,
}

impl MarketState {
    pub fn is_clearing(&self) -> bool {
        self.stage == CLEARING_STAGE
    }
}

// AcccountId, quantity, price
pub type Submission = (Account, u64, u64);

pub type Account = Vec<u8>;

impl ParachainClient {
    pub async fn new(rpc_url: &str) -> Result<Self> {
        let rpc_client = WsClientBuilder::default()
            .build(rpc_url)
            .await
            .map_err(|e| anyhow!("failed to build rpc client, err: {e:?}"))?;
        Ok(ParachainClient { rpc_client })
    }

    pub async fn get_market_state(&self) -> Result<MarketState> {
        self.rpc_client
            .request("marketState_getSubmissions", rpc_params![])
            .await
            .map_err(|e| anyhow!("failed to get storage value, err: {:?}", e))
    }
}
