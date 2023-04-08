use anyhow::{anyhow, Result};
use jsonrpsee::rpc_params;
use jsonrpsee::{
    core::client::ClientT,
    ws_client::{WsClient, WsClientBuilder},
};
use serde::{Deserialize, Serialize};
use codec::{Encode, Decode};

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

    async fn get_storage_keys(&self, prefix: &str) -> Result<Vec<StorageKey>> {
        const PER_PAGE: u32 = 100;
        let encoded_prefix = prefix.encode();
        let encoded_count = PER_PAGE.encode();
        let mut all_keys = Vec::new();
        let mut start_key: Option<StorageKey> = None;
        loop {
            let start_key_encoded: Option<Vec<u8>> = start_key.map(|k| k.encode());
            let mut keys: Vec<StorageKey> = self.rpc_client
                .request("state_getStorageKeysPaged", rpc_params![&encoded_prefix, &encoded_count, start_key_encoded])
                .await
                .map_err(|e| anyhow!("failed to get market state, err: {:?}", e))?;
            if keys.is_empty() {
                break
            }
            start_key = Some(keys[0].clone());
            all_keys.append(&mut keys);
        }
        Ok(all_keys)
    }
}

// https://docs.rs/sp-storage/latest/sp_storage/struct.StorageKey.html
type StorageKey = Vec<u8>;