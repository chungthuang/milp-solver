pub mod error;

use std::ops::Mul;

use crate::error::Error;
use good_lp::solvers::lp_solvers::LpSolution;
use good_lp::{
    solvers::lp_solvers::{CbcSolver, LpSolver},
    variable, Constraint, Expression, ProblemVariables, Solution, SolverModel, Variable,
};
use log::debug;
use parachain_client::{AccountId, FlexibleProduct, MarketSolution, Product};
use uuid::Uuid;

const DECISION_ACCEPTED: f64 = 1.;

#[derive(Copy, Clone)]
enum ProductType {
    Bid,
    Ask,
}

pub fn solve(
    bids: Vec<(AccountId, Vec<FlexibleProduct>)>,
    asks: Vec<(AccountId, Vec<FlexibleProduct>)>,
    periods: u32,
) -> Result<MarketSolution, Error> {
    let mut vars = ProblemVariables::new();

    let bid_formulation = formulate(&mut vars, &bids, periods as usize);
    let ask_formulation = formulate(&mut vars, &asks, periods as usize);

    let social_welfare: Expression = bid_formulation.welfare.into_iter().sum::<Expression>()
        - ask_formulation.welfare.into_iter().sum::<Expression>();
    debug!("social welfare {:?}", social_welfare);

    let problem_id = Uuid::new_v4();
    debug!("Problem ID {:?}", problem_id);
    let cbc_solver =
        LpSolver(CbcSolver::new().with_temp_solution_file(format!("/tmp/milp_{}.sol", problem_id)));

    let mut model = vars.maximise(social_welfare).using(cbc_solver);

    for (period, (bids_quantities, asks_quantities)) in bid_formulation
        .quantities
        .into_iter()
        .zip(ask_formulation.quantities)
        .enumerate()
    {
        let quantity_match: Expression = bids_quantities.into_iter().sum::<Expression>()
            - asks_quantities.into_iter().sum::<Expression>();
        debug!(
            "Period {}: quantity match constraint {:?}",
            period, quantity_match
        );
        model = model.with(quantity_match.eq(0));
    }

    for constraint in bid_formulation.flex_load_constraints.into_iter() {
        model = model.with(constraint);
    }
    for constraint in ask_formulation.flex_load_constraints.into_iter() {
        model = model.with(constraint);
    }

    let sol = model
        .solve()
        .map_err(|err| Error::Solver(err.to_string()))?;

    evaluate(
        sol,
        bids,
        asks,
        bid_formulation.decisions,
        ask_formulation.decisions,
        periods as usize,
    )
}

struct ProblemFormulation {
    decisions: Vec<Variable>,
    flex_load_constraints: Vec<Constraint>,
    // A vector of bid/ask quantities for each period
    quantities: Vec<Vec<Expression>>,
    // Utility/cost
    welfare: Vec<Expression>,
}

/// Add decision variables for products to vars, return the decision variables
fn formulate(
    vars: &mut ProblemVariables,
    products: &[(AccountId, Vec<FlexibleProduct>)],
    periods: usize,
) -> ProblemFormulation {
    let mut decisions: Vec<Variable> = Vec::new();
    let mut flex_load_constraints: Vec<Constraint> = Vec::new();
    let mut quantities: Vec<Vec<Expression>> = Vec::with_capacity(periods as usize);
    for _ in 0..periods {
        quantities.push(Vec::new());
    }
    let mut welfare: Vec<Expression> = Vec::new();
    for (_account, flexible_products) in products.iter() {
        for flexible_product in flexible_products.iter() {
            let mut flex_load_decisions =
                vars.add_vector(variable().binary(), flexible_product.len());
            flex_load_constraints.push(flex_load_decisions.iter().sum::<Expression>().leq(1));
            for (product, decision) in flexible_product.iter().zip(flex_load_decisions.iter()) {
                for period in product.start_period..product.end_period {
                    quantities[period as usize].push(decision.mul(product.quantity as f64));
                    welfare.push(decision.mul((product.quantity * product.price) as f64));
                }
            }
            decisions.append(&mut flex_load_decisions);
        }
    }
    ProblemFormulation {
        decisions,
        flex_load_constraints,
        quantities,
        welfare,
    }
}

fn evaluate(
    sol: LpSolution,
    bids: Vec<(AccountId, Vec<FlexibleProduct>)>,
    asks: Vec<(AccountId, Vec<FlexibleProduct>)>,
    bid_status: Vec<Variable>,
    ask_status: Vec<Variable>,
    periods: usize,
) -> Result<MarketSolution, Error> {
    let bids_solutions =
        evaluate_product_decisions(&sol, bids, bid_status, ProductType::Bid, periods)?;
    let asks_solutions =
        evaluate_product_decisions(&sol, asks, ask_status, ProductType::Ask, periods)?;

    let mut auction_prices = Vec::with_capacity(periods);
    for (period, (min_bid_price, max_ask_price)) in bids_solutions
        .auction_prices
        .iter()
        .zip(asks_solutions.auction_prices)
        .enumerate()
    {
        debug!(
            "Period {}: Min bid price {}, max ask price at period {}",
            period, min_bid_price, max_ask_price
        );
        if *min_bid_price < max_ask_price {
            return Err(Error::InvalidSolution(format!(
                "Period {}: Minimum bid price {} is less than maximum ask price {}",
                period, min_bid_price, max_ask_price
            )));
        }
        auction_prices.push(*min_bid_price);
    }

    for (period, (bid_quantity, ask_quantity)) in bids_solutions
        .quantities
        .iter()
        .zip(asks_solutions.quantities)
        .enumerate()
    {
        if *bid_quantity != ask_quantity {
            return Err(Error::InvalidSolution(format!(
                "Period {}: Total bid quantity {} != total ask quantity {}",
                period, bid_quantity, ask_quantity
            )));
        }
    }

    Ok(MarketSolution {
        bids: bids_solutions.solutions,
        asks: asks_solutions.solutions,
        auction_prices,
    })
}

struct ProductDecisions {
    solutions: Vec<(AccountId, Vec<Option<Product>>)>,
    // Auction price by period
    auction_prices: Vec<u64>,
    // Quantity by period
    quantities: Vec<u64>,
}

/// Decide if a product is accepted, and if so, which flexible load
fn evaluate_product_decisions(
    sol: &LpSolution,
    products: Vec<(AccountId, Vec<FlexibleProduct>)>,
    decisions: Vec<Variable>,
    product_type: ProductType,
    periods: usize,
) -> Result<ProductDecisions, Error> {
    let mut solutions: Vec<(AccountId, Vec<Option<Product>>)> = Vec::with_capacity(products.len());
    let mut auction_prices: Vec<u64> = Vec::with_capacity(periods);
    let mut quantities: Vec<u64> = Vec::with_capacity(periods);
    for _ in 0..periods {
        auction_prices.push(0);
        quantities.push(0);
    }

    let mut decisions = decisions.into_iter();
    // Iterate the same way as solve creates the vector of bids
    for (account, flexible_products) in products.into_iter() {
        let mut account_solutions: Vec<Option<Product>> =
            Vec::with_capacity(flexible_products.len());
        for flexible_product in flexible_products.iter() {
            let mut accepted: Option<Product> = None;
            for product in flexible_product.iter() {
                let Some(decision) = decisions.next() else {
                    return Err(Error::InvalidSolution("More products than decision variables".to_owned()));
                };
                if sol.value(decision) == DECISION_ACCEPTED {
                    // Make sure at most only 1 flexible load is accepted
                    if accepted.is_some() {
                        return Err(Error::InvalidSolution(
                            "Multiple flexible load accepted".to_owned(),
                        ));
                    }
                    accepted = Some(*product);
                    for period in product.start_period..product.end_period {
                        let period = period as usize;
                        if auction_prices[period] == 0 {
                            auction_prices[period] = product.price;
                        } else {
                            match product_type {
                                ProductType::Bid => {
                                    if product.price < auction_prices[period] {
                                        auction_prices[period] = product.price;
                                    }
                                }
                                ProductType::Ask => {
                                    if product.price > auction_prices[period] {
                                        auction_prices[period] = product.price;
                                    }
                                }
                            };
                        }
                        quantities[period] += product.quantity;
                    }
                }
            }
            account_solutions.push(accepted);
        }
        solutions.push((account, account_solutions));
    }
    if decisions.next().is_some() {
        return Err(Error::InvalidSolution(
            "More decision variables thatn products".to_owned(),
        ));
    }
    Ok(ProductDecisions {
        solutions,
        auction_prices,
        quantities,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use test_log::test;

    // Evaluate solution for single period products
    #[test]
    fn test_solve_single_products() {
        let account_1 = test_account(1);
        let account_2 = test_account(2);
        let account_3 = test_account(3);

        let bid_1 = Product {
            price: 6,
            quantity: 5,
            start_period: 1,
            end_period: 2,
        };
        let ask_1 = Product {
            price: 5,
            quantity: 5,
            start_period: 1,
            end_period: 2,
        };

        let bid_2 = Product {
            price: 8,
            quantity: 2,
            start_period: 3,
            end_period: 4,
        };
        let ask_2 = Product {
            price: 7,
            quantity: 2,
            start_period: 3,
            end_period: 4,
        };

        let bids = vec![(account_1, vec![fixed_load(bid_1), fixed_load(bid_2)])];

        let asks = vec![
            (account_2, vec![fixed_load(ask_1)]),
            (account_3, vec![fixed_load(ask_2)]),
        ];

        let solution = solve(bids, asks, 5).unwrap();
        assert_eq!(
            solution.bids,
            vec![(account_1, vec![Some(bid_1), Some(bid_2)])]
        );
        assert_eq!(
            solution.asks,
            vec![
                (account_2, vec![Some(ask_1)]),
                (account_3, vec![Some(ask_2)])
            ]
        );
        assert_eq!(
            solution.auction_prices,
            vec![0, bid_1.price, 0, bid_2.price, 0]
        )
    }

    #[test]
    fn test_solve_single_products_max_social_welfare() {
        let account_1 = test_account(1);
        let account_2 = test_account(2);
        let account_3 = test_account(3);

        let bid_1 = Product {
            price: 6,
            quantity: 5,
            start_period: 1,
            end_period: 2,
        };
        let ask_1 = Product {
            price: 6,
            quantity: 5,
            start_period: 1,
            end_period: 2,
        };

        let bid_2 = Product {
            price: 7,
            quantity: 5,
            start_period: 1,
            end_period: 2,
        };
        let ask_2 = Product {
            price: 7,
            quantity: 5,
            start_period: 1,
            end_period: 2,
        };

        let bids = vec![(account_1, vec![fixed_load(bid_1), fixed_load(bid_2)])];

        let asks = vec![
            (account_2, vec![fixed_load(ask_1)]),
            (account_3, vec![fixed_load(ask_2)]),
        ];

        // bid_2 will match with ask_1, because the social welfare would be 7 * 5 - 6 * 5 = 5
        // if bid_1 matches with ask_1 and bid_2 matches with ask_2, than the social welfare would be 0
        let solution = solve(bids, asks, 3).unwrap();
        assert_eq!(solution.bids, vec![(account_1, vec![None, Some(bid_2)])]);
        assert_eq!(
            solution.asks,
            vec![(account_2, vec![Some(ask_1)]), (account_3, vec![None])]
        );
        assert_eq!(solution.auction_prices, vec![0, bid_2.price, 0])
    }

    #[test]
    fn test_solve_continuous_products() {
        let account_1 = test_account(1);
        let account_2 = test_account(2);
        let account_3 = test_account(3);

        let bid_1 = Product {
            price: 5,
            quantity: 5,
            start_period: 1,
            end_period: 3,
        };
        let bid_2 = Product {
            price: 7,
            quantity: 2,
            start_period: 5,
            end_period: 7,
        };

        let ask_1 = Product {
            price: 4,
            quantity: 5,
            start_period: 1,
            end_period: 3,
        };
        let ask_2 = Product {
            price: 6,
            quantity: 2,
            start_period: 5,
            end_period: 7,
        };

        let bids = vec![(account_1, vec![fixed_load(bid_1), fixed_load(bid_2)])];

        let asks = vec![
            (account_2, vec![fixed_load(ask_1)]),
            (account_3, vec![fixed_load(ask_2)]),
        ];

        let solution = solve(bids, asks, 7).unwrap();
        assert_eq!(
            solution.bids,
            vec![(account_1, vec![Some(bid_1), Some(bid_2)])]
        );
        assert_eq!(
            solution.asks,
            vec![
                (account_2, vec![Some(ask_1)]),
                (account_3, vec![Some(ask_2)])
            ]
        );
        assert_eq!(
            solution.auction_prices,
            vec![0, bid_1.price, bid_1.price, 0, 0, bid_2.price, bid_2.price]
        )
    }

    #[test]
    fn test_solve_overlapping_continuous_products() {
        let account_1 = test_account(1);
        let account_2 = test_account(2);
        let account_3 = test_account(3);

        let bid_1 = Product {
            price: 5,
            quantity: 5,
            start_period: 1,
            end_period: 4,
        };
        let bid_2 = Product {
            price: 7,
            quantity: 2,
            start_period: 2,
            end_period: 5,
        };

        let ask_1 = Product {
            price: 4,
            quantity: 7,
            start_period: 1,
            end_period: 5,
        };

        let bids = vec![(account_1, vec![fixed_load(bid_1), fixed_load(bid_2)])];

        let asks = vec![(account_2, vec![fixed_load(ask_1.clone())])];

        let solution = solve(bids, asks.clone(), 5).unwrap();
        // No enough bid quantity to match ask_1 at period 1
        assert_no_solution(solution);

        let bid_3 = Product {
            price: 5,
            quantity: 2,
            start_period: 1,
            end_period: 2,
        };
        let bid_4 = Product {
            price: 6,
            quantity: 5,
            start_period: 4,
            end_period: 5,
        };

        let bids = vec![
            (account_1, vec![fixed_load(bid_1), fixed_load(bid_2)]),
            (account_3, vec![fixed_load(bid_3), fixed_load(bid_4)]),
        ];

        let solution = solve(bids.clone(), asks, 5).unwrap();
        assert_eq!(
            solution.bids,
            vec![
                (account_1, vec![Some(bid_1), Some(bid_2)]),
                (account_3, vec![Some(bid_3), Some(bid_4)])
            ]
        );
        assert_eq!(solution.asks, vec![(account_2, vec![Some(ask_1)]),]);
        assert_eq!(solution.auction_prices, vec![0, 5, 5, 5, 6])
    }

    #[test]
    fn test_solve_flexible_products() {
        let account_1 = test_account(1);
        let account_2 = test_account(2);
        let account_3 = test_account(3);

        let bid_1 = vec![
            Product {
                price: 5,
                quantity: 5,
                start_period: 1,
                end_period: 3,
            },
            Product {
                price: 3,
                quantity: 3,
                start_period: 2,
                end_period: 4,
            },
        ];
        let bid_2 = vec![
            Product {
                price: 7,
                quantity: 2,
                start_period: 2,
                end_period: 4,
            },
            Product {
                price: 8,
                quantity: 2,
                start_period: 5,
                end_period: 7,
            },
        ];

        let ask_1 = vec![Product {
            price: 6,
            quantity: 2,
            start_period: 2,
            end_period: 4,
        }];
        let ask_2 = vec![Product {
            price: 2,
            quantity: 3,
            start_period: 1,
            end_period: 5,
        }];

        let bids = vec![(account_1, vec![bid_1, bid_2.clone()])];

        let asks = vec![(account_2, vec![ask_1.clone()]), (account_3, vec![ask_2])];

        let solution = solve(bids, asks, 7).unwrap();
        assert_eq!(solution.bids, vec![(account_1, vec![None, Some(bid_2[0])])]);
        assert_eq!(
            solution.asks,
            vec![(account_2, vec![Some(ask_1[0])]), (account_3, vec![None])]
        );
        assert_eq!(solution.auction_prices, vec![0, 0, 7, 7, 0, 0, 0])
    }

    fn test_account(account_index: u8) -> AccountId {
        [account_index; 32]
    }

    fn fixed_load(product: Product) -> FlexibleProduct {
        vec![product]
    }

    fn assert_no_solution(sol: MarketSolution) {
        for (_, bids) in sol.bids {
            for b in bids {
                assert!(b.is_none());
            }
        }
        for (_, asks) in sol.asks {
            for a in asks {
                assert!(a.is_none());
            }
        }
        for price in sol.auction_prices {
            assert_eq!(price, 0);
        }
    }
}
