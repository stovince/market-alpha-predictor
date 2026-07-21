<<<<<<< HEAD
# market-alpha-predictor

## 🚀 Project Oversight
Deep learning project built with TensorFlow, aimed at forecasting short-term stock price movements.

## Main Objectives
- Learn to apply deep learning to finanical time series.
- Incorporate technical indicators to enrich input singals beyond raw prices.
  
=======
# Market Alpha Predictor

This project is an educational, reproducible research workflow for short-term stock-direction prediction using Python and TensorFlow. It does not constitute financial advice or proof of a consistently profitable trading strategy.

## Objective
Predict next-day stock price direction from historical OHLCV data and engineered technical indicators.

## Dataset
Raw daily stock data is loaded from `data/raw/{symbol}.csv` with columns including `Open`, `High`, `Low`, `Close`, and `Volume`.

## Prediction Target
The model predicts `target_direction`, a binary label indicating whether the next days return is positive. This is preferred because it focuses on directional forecasting and avoids claiming absolute price prediction.

## Features and Indicators
Features are computed only from data available at time t, without using future information.
The pipeline generates:
- raw returns and log returns
- moving averages (MA)
- exponential moving averages (EMA)
- rolling volatility
- RSI
- MACD and MACD signal/histogram
- volume change and volume moving average

## Model Architecture
A compact TensorFlow/Keras MLP is used:
- dense layers: 128 -> 64 -> 32
- dropout layers for regularization
- sigmoid output for binary classification
- binary cross-entropy loss

## Train / Validation / Test Split
Data is split chronologically into train, validation, and test sets. No random shuffling is performed before splitting. This prevents look-ahead bias.

## Baseline
A simple baseline classifier that predicts the most frequent direction is included so the model can be compared against a naive benchmark.

## Evaluation Metrics
For classification, the pipeline reports:
- accuracy
- precision
- recall
- F1 score
- confusion matrix

## Installation
Create and activate a Python virtual environment, then install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Usage
Prepare data:

```powershell
python src\prepare_data.py --symbol AAPL
```

Train model:

```powershell
python src\train.py --symbol AAPL
```

Evaluate model:

```powershell
python src\evaluate.py --symbol AAPL
```

## Repository Structure
- `data/`: raw and processed data
- `indicators/`: existing indicator helper module
- `src/`: data preparation, model, training, and evaluation scripts
- `results/`: saved evaluation outputs and plots
- `models/`: saved TensorFlow model artifacts
- `tests/`: basic tests for data processing

## Limitations
- Only daily historical data is used.
- This is a research pipeline, not a deployed trading system.
- The model does not include transaction costs or slippage in evaluation.
- Results are not proof of profitability.
>>>>>>> feature/tf-research-pipeline
