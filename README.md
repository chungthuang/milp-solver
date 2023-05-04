This project provides a web server to interact with a linear programming solver. 

# Prerequisite
- Install [CBC](https://www.coin-or.org/Cbc/) library.
- Install [subxt](https://github.com/paritytech/subxt).

# Update parachain metadata
1. Make sure the parachain is running
2. `subxt metadata -f bytes --url http://localhost:8844 > metadata/parachain.scale`