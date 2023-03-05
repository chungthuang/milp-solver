use solver::Solver;

#[tokio::main]
async fn main() {
    let mut handles = vec![];
    handles.push(tokio::spawn(async {
        tokio::spawn(poll_market_state());
    }));
    handles.push(tokio::spawn(poll_solution()));

    futures::future::join_all(handles).await;
}

async fn poll_market_state() {

}

async fn poll_solution() {

}
