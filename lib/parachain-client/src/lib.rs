use anyhow::{anyhow, Result};
use jsonrpsee::rpc_params;
use jsonrpsee::{
    core::client::ClientT,
    ws_client::{WsClient, WsClientBuilder},
};
use serde::{Deserialize, Serialize};
//use codec::{Encode, Decode};

pub struct ParachainClient {
    rpc_client: WsClient,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct MarketSubmissions {
    pub bids: Vec<(u64, u64)>,
    pub asks: Vec<(u64, u64)>,
    pub stage: u64,
}

impl ParachainClient {
    pub async fn new(rpc_url: &str) -> Result<Self> {
        let rpc_client = WsClientBuilder::default()
            .build(rpc_url)
            .await
            .map_err(|e| anyhow!("failed to build rpc client, err: {:?}", e))?;
        Ok(ParachainClient { rpc_client })
    }

    pub async fn get_market_state(&self) -> Result<MarketSubmissions> {
        self.rpc_client
            .request("state_getStorage", rpc_params![])
            .await
            .map_err(|e| anyhow!("failed to get market state, err: {:?}", e))
    }
}
