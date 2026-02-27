# Deep Learning for Rough Volatility Models

This repository houses a quantitative finance project focused on resolving the curse of dimensionality in pricing and hedging under rough volatility models (e.g., rough Bergomi).

It benchmarks state-of-the-art Deep Learning methods (Deep BSDEs / PINNs) against standard baselines and challenge datasets, with a strong emphasis on highly optimized computation and 3D visualization of volatility surfaces and hedging errors.

## Structure
- `src/`: Core Python modules for fractional Brownian motion, option pricing, baselines, and PyTorch models.
- `notebooks/`: Jupyter notebooks for mathematical exploration, 3D plotting, and validation.
- `tests/`: Unit tests to verify the statistical integrity of stochastic processes.
