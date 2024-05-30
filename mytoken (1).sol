// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract TokenPricePredictor {
    struct Prediction {
        address predictor;
        uint256 day;
        uint256 prediction;
        string signature;
        uint256 timestamp;
    }

    Prediction[] public predictions;

    event PredictionStored(address indexed predictor, uint256 day, uint256 prediction, string signature, uint256 timestamp);

    function storePrediction(address _predictor, uint256 _day, uint256 _prediction, string memory _signature) public {
        predictions.push(Prediction({
            predictor: _predictor,
            day: _day,
            prediction: _prediction,
            signature: _signature,
            timestamp: block.timestamp
        }));
        emit PredictionStored(_predictor, _day, _prediction, _signature, block.timestamp);
    }

    function getPredictions() public view returns (Prediction[] memory) {
        return predictions;
    }
}
