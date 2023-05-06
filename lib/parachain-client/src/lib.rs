use anyhow::{anyhow, Result};
use jsonrpsee::{
    core::client::ClientT,
    http_client::{HttpClient, HttpClientBuilder},
    rpc_params,
};
use parachain::runtime_types::{
    market_state::pallet::OperatingPeriods as ParachainOperatingPeriods,
    sp_core::bounded::bounded_vec::BoundedVec,
};
use serde::Deserialize;
use sp_keyring::{sr25519::sr25519, AccountKeyring};
use subxt::utils::H256;
use subxt::{config::PolkadotConfig, tx::PairSigner, utils::AccountId32, OnlineClient};

const CLEARING_STAGE: Stage = 1;

pub struct ParachainClient {
    parachain_api: OnlineClient<PolkadotConfig>,
    signer: PairSigner<PolkadotConfig, sr25519::Pair>,
    rpc_client: HttpClient,
}

type Stage = u64;

#[derive(Debug, Deserialize)]
pub struct MarketState {
    // Can't deserialize directly to AccountId32 because it's a tuple of [u8; 32]
    pub bids: Vec<(AccountId, Vec<Product>)>,
    pub asks: Vec<(AccountId, Vec<Product>)>,
    pub stage: Stage,
    pub periods: u32,
}

impl MarketState {
    pub fn is_clearing(&self) -> bool {
        self.stage == CLEARING_STAGE
    }
}

#[derive(Debug)]
pub struct MarketSolution {
    // For each account, track if the bids are accepted
    pub bids: Vec<(AccountId, Vec<Option<OperatingPeriods>>)>,
    // For each account, track if the asks are accepted
    pub asks: Vec<(AccountId, Vec<Option<OperatingPeriods>>)>,
    // Auction price for each period
    pub auction_prices: Vec<u64>,
}

// Re-export for solver crate
pub type AccountId = [u8; 32];

#[derive(Clone, Debug, Default, Eq, PartialEq, Deserialize)]
pub struct Product {
    pub price: u64,
    pub quantity: u64,
    pub flexible_loads: Vec<OperatingPeriods>,
}

#[derive(Copy, Clone, Debug, Default, Eq, PartialEq, Deserialize)]
pub struct OperatingPeriods {
    pub start: u32,
    // A single product will have start == end
    pub end: u32,
}

impl Into<ParachainOperatingPeriods> for OperatingPeriods {
    fn into(self) -> ParachainOperatingPeriods {
        ParachainOperatingPeriods {
            start: self.start,
            end: self.end,
        }
    }
}

impl ParachainClient {
    pub async fn new(rpc_addr: &str) -> Result<Self> {
        let parachain_api = OnlineClient::from_url(format!("ws://{rpc_addr}"))
            .await
            .map_err(|e| anyhow!("failed to build subxt client, err: {e:?}"))?;
        let signer = PairSigner::new(AccountKeyring::Alice.pair());
        let rpc_client = HttpClientBuilder::default()
            .build(format!("http://{rpc_addr}"))
            .map_err(|e| anyhow!("failed to build rpc client, err: {e:?}"))?;
        Ok(ParachainClient {
            parachain_api,
            signer,
            rpc_client,
        })
    }

    pub async fn get_market_state(&self) -> Result<MarketState> {
        self.rpc_client
            .request("marketState_getSubmissions", rpc_params![])
            .await
            .map_err(|e| anyhow!("failed to get storage value, err: {:?}", e))
    }

    pub async fn submit_solution(&self, solution: MarketSolution) -> Result<H256> {
        let tx = parachain::tx().market_state().submit_solution(
            BoundedVec(solution.auction_prices),
            BoundedVec(
                solution
                    .bids
                    .into_iter()
                    .map(|(a, operating_periods)| {
                        (
                            AccountId32(a),
                            BoundedVec(
                                operating_periods
                                    .into_iter()
                                    .map(|op| op.map(|op| op.into()))
                                    .collect(),
                            ),
                        )
                    })
                    .collect(),
            ),
            BoundedVec(
                solution
                    .asks
                    .into_iter()
                    .map(|(a, operating_periods)| {
                        (
                            AccountId32(a),
                            BoundedVec(
                                operating_periods
                                    .into_iter()
                                    .map(|op| op.map(|op| op.into()))
                                    .collect(),
                            ),
                        )
                    })
                    .collect(),
            ),
        );
        let hash = self
            .parachain_api
            .tx()
            .sign_and_submit_default(&tx, &self.signer)
            .await?;
        Ok(hash)
    }
}

#[subxt::subxt(runtime_metadata_path = "../../metadata/parachain.scale")]
mod parachain {}

#[cfg(test)]
mod tests {
    use super::*;

    fn get_market_state() -> Result<MarketState> {
        Ok(MarketState {
            bids: vec![(AccountId([1; 32]), 10, 7), (AccountId([2; 32]), 10, 6)],
            asks: vec![(AccountId([3; 32]), 10, 5)],
            stage: CLEARING_STAGE,
        })
    }
}
