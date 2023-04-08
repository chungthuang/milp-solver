use anyhow::{anyhow, Result};
use byteorder::{ByteOrder, LittleEndian};
use codec::{Decode, Encode};
use core::hash::Hasher;
use digest::Digest;
use jsonrpsee::rpc_params;
use jsonrpsee::{
    core::client::ClientT,
    ws_client::{WsClient, WsClientBuilder},
};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use twox_hash;

const PALLET_NAME: &str = "MarketInputPallet"; // "MarketState";
const CLEARING_STAGE: Stage = 1;

pub struct ParachainClient {
    rpc_client: WsClient,
}

type Stage = u64;

#[derive(Debug)]
pub struct MarketState {
    pub bids: Vec<Submission>,
    pub asks: Vec<Submission>,
}

#[derive(Debug)]
pub struct Submission {
    pub account: Account,
    pub quantity_price: QuantityPrice,
}

pub type Account = Vec<u8>;

pub type QuantityPrice = (u64, u64);

impl ParachainClient {
    pub async fn new(rpc_url: &str) -> Result<Self> {
        let rpc_client = WsClientBuilder::default()
            .build(rpc_url)
            .await
            .map_err(|e| anyhow!("failed to build rpc client, err: {e:?}"))?;
        Ok(ParachainClient { rpc_client })
    }

    pub async fn get_market_state(&self) -> Result<MarketState> {
        let stage = self.get_stage().await?;
        if stage != CLEARING_STAGE {
            return Err(anyhow!(
                "Should not get market state during stage {stage:?}"
            ));
        }
        let bids = self.get_submissions("Bids").await?;
        let asks = self.get_submissions("Asks").await?;
        Ok(MarketState { bids, asks })
    }

    async fn get_stage(&self) -> Result<Stage> {
        let key = self.storage_value_key("Stage");
        let value: Vec<u8> = self.get_storage_value(&key).await?;
        Stage::decode(&mut value.as_ref())
            .map_err(|e| anyhow!("Failed to decode storage value into stage, err: {e:?}"))
    }

    async fn get_submissions(&self, storage_name: &str) -> Result<Vec<Submission>> {
        let keys = self.get_storage_keys(storage_name).await?;
        let mut submissions = Vec::with_capacity(keys.len());
        for key in keys.into_iter() {
            // TODO: Add retry
            let quantity_price = self.get_storage_value(&key).await?;
            submissions.push(Submission {
                account: key,
                quantity_price,
            });
        }
        Ok(submissions)
    }

    async fn get_storage_keys(&self, prefix: &str) -> Result<Vec<StorageKey>> {
        const PER_PAGE: u32 = 100;
        // The key format for storage map has the same prefix as the key format for storage value
        let encoded_prefix = self.storage_value_key(prefix);
        println!("ecoded prefix {:?}", encoded_prefix);
        let encoded_count = PER_PAGE.encode();
        let mut all_keys = Vec::new();
        let mut start_key: Option<StorageKey> = None;
        loop {
            let start_key_encoded: Option<Vec<u8>> = start_key.map(|k| k.encode());
            let mut keys: Vec<StorageKey> = self
                .rpc_client
                .request(
                    "state_getStorageKeysPaged",
                    rpc_params![&encoded_prefix, &encoded_count, start_key_encoded],
                )
                .await
                .map_err(|e| anyhow!("failed to get storage key, err: {:?}", e))?;
            if keys.is_empty() {
                break;
            }
            start_key = Some(keys[0].clone());
            all_keys.append(&mut keys);
        }
        Ok(all_keys)
    }

    async fn get_storage_value<T: DeserializeOwned>(&self, key: &StorageKey) -> Result<T> {
        self.rpc_client
            .request("state_getStorage", rpc_params!["0xfe7737a7a8ac4c0d38d011f5667b70210xbd08734cf8008496dbbf777aed2138f2"])
            .await
            .map_err(|e| anyhow!("failed to get storage value, err: {:?}", e))
    }

    fn storage_value_key(&self, storage_name: &str) -> StorageKey {
        // The key format for storage value is:
        // xxhash128("ModuleName") + xxhash128("StorageName")
        [twox_128(PALLET_NAME.as_bytes()), twox_128(storage_name.as_bytes())].concat()
    }
}

// https://docs.rs/sp-storage/latest/sp_storage/struct.StorageKey.html
type StorageKey = Vec<u8>;

/// Do a XX 128-bit hash and place result in `dest`.
fn twox_128_into(data: &[u8], dest: &mut [u8; 16]) {
    let r0 = twox_hash::XxHash::with_seed(0).chain_update(data).finish();
    let r1 = twox_hash::XxHash::with_seed(1).chain_update(data).finish();
    LittleEndian::write_u64(&mut dest[0..8], r0);
    LittleEndian::write_u64(&mut dest[8..16], r1);
}

/// Do a XX 128-bit hash and return result.
pub(crate) fn twox_128(data: &[u8]) -> [u8; 16] {
    let mut r: [u8; 16] = [0; 16];
    twox_128_into(data, &mut r);
    r
}