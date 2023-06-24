use anyhow::{anyhow, Result};
use jsonrpsee::{
    core::client::ClientT,
    http_client::{HttpClient, HttpClientBuilder},
    rpc_params,
};
use parachain::runtime_types::sp_core::bounded::bounded_vec::BoundedVec;
use serde::Deserialize;
use sp_keyring::{sr25519::sr25519, AccountKeyring};
use subxt::utils::H256;
use subxt::{config::PolkadotConfig, tx::PairSigner, OnlineClient};

const CLEARING_STAGE: Stage = 1;

pub struct ParachainClient {
    parachain_api: OnlineClient<PolkadotConfig>,
    signer: PairSigner<PolkadotConfig, sr25519::Pair>,
    rpc_client: HttpClient,
}

pub type ProductId = u32;

pub type SelectedFlexibleLoad = u32;

type Stage = u64;

#[derive(Debug, Deserialize)]
pub struct MarketState {
    pub bids: Vec<(ProductId, FlexibleProduct)>,
    pub asks: Vec<(ProductId, FlexibleProduct)>,
    pub stage: u64,
    pub periods: u32,
    pub grid_price: u64,
    pub feed_in_tariff: u64,
}

impl MarketState {
    pub fn is_clearing(&self) -> bool {
        self.stage == CLEARING_STAGE
    }
}

#[derive(Debug)]
pub struct MarketSolution {
    // For each account, track if the bids are accepted
    pub bids: Vec<AcceptedProduct>,
    // For each account, track if the asks are accepted
    pub asks: Vec<AcceptedProduct>,
    // Auction price for each period
    pub auction_prices: Vec<u64>,
}

impl MarketSolution {
    pub fn no_solution(&self) -> bool {
        if !self.bids.is_empty() {
            return false;
        }
        if !self.asks.is_empty() {
            return false;
        }
        for p in self.auction_prices.iter() {
            if *p > 0 {
                return false;
            }
        }
        true
    }
}

#[derive(Copy, Clone, Debug, Default, Eq, PartialEq, Deserialize)]
pub struct Product {
    pub price: u64,
    pub quantity: u64,
    pub start_period: u32,
    pub end_period: u32,
    pub can_partially_accept: bool,
}

#[derive(Copy, Clone, Debug, Default, Eq, PartialEq, Deserialize)]
pub struct AcceptedProduct {
    pub id: ProductId,
    pub load_index: SelectedFlexibleLoad,
    // Percentage [0, 100] of quantity accepted from start to end period.
    // We use an integer here because the runtime cannot perform floating point arithmetic.
    pub percentage: u8,
}

impl Into<parachain::runtime_types::market_state::pallet::AcceptedProduct> for AcceptedProduct {
    fn into(self) -> parachain::runtime_types::market_state::pallet::AcceptedProduct {
        parachain::runtime_types::market_state::pallet::AcceptedProduct {
            id: self.id,
            load_index: self.load_index,
            percentage: self.percentage,
        }
    }
}

pub type FlexibleProduct = Vec<Product>;

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
            BoundedVec(solution.bids.into_iter().map(|b| b.into()).collect()),
            BoundedVec(solution.asks.into_iter().map(|a| a.into()).collect()),
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
