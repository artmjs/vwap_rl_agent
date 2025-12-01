# VWAP Optimal Execution (Synthetic SHDP Baseline)

## Overview

This project implements a simplified optimal VWAP execution framework based on a stochastic control and SHDP approach proposed by Enzo Busseti and Stephen Boyd in "Volume Weighted Average Price Optimal Execution" (2015).

The current version works on synthetic intraday data (prices, volumes, spreads). The code is structured so that real intraday data (e.g. AAPL, EURUSD) can be used with a different data loader.

## Method

- **Volume model**: intraday lognormal volume profile, estimated from a rolling window of past days.
- **Price-risk model**: per-minute return variance `sigma2` from rolling returns (Gaussian / Student-t).
- **Baselines**:
  - Static VWAP schedule from historical intraday profile \\(\hat{\pi}_t\\).
  - SHDP controller: closed-loop policy with a tracking gap feedback term and a price-variance penalty.

Objective: minimize

\\[
J = S + \lambda R,
\\]

where `S` is VWAP slippage and `R` is a risk / tracking penalty.

## Repo structure

High-level layout:

```text
src/
  baselines/
    vwap_static.py       # static VWAP baseline: pi_hat estimation and schedule
    shdp_controller.py   # SHDPController (closed-loop feedback policy)
  sim/
    synthetic.py         # synthetic price/volume/spread generator
    env.py               # VWAP execution environment
  exp/
    run_baselines.py     # main experiment script (Static vs SHDP on synthetic data)
    evaluate.py          # open/closed-loop evaluation (S, R, J, tracking error G)

```

## How to run

```bash
# from repo root
python -m src.exp.run_baselines
```
