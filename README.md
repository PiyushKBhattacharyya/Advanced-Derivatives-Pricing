# ⚡ Deep BSDE & Frictional Reinforcement Learning for Non-Markovian Hedging

This repository houses a comprehensive, PhD-level quantitative finance architecture engineered to resolve the **curse of dimensionality** in pricing options under rough volatility models (e.g., fractional Brownian motion).

By merging deep PyTorch Neural Networks with **Proximal Policy Optimization (PPO)** Reinforcement Learning, this framework practically bypasses rigid legacy banking methods (like SABR and Black-Scholes), offering live empirical execution boundaries on the S&P 500 Index.

## 🌟 Core Architecture Components

### 1. Live PyTorch Pricing Engine (Deep BSDE)
Instead of numerically solving a Partial Differential Equation (PDE), the architecture translates option pricing into a **Backward Stochastic Differential Equation (BSDE)**.
*   An **LSTM Encoder** physically compresses non-Markovian historical rough volatility paths into actionable Markovian states.
*   Calculates a dense 3D Volatility Pricing Surface (`Strike K` vs `Maturity T`), visualizing exactly where standard banking lines historically mathematically collapse.

### 2. $2^{nd}$-Order Neural Curvature Extractors (Gamma $\frac{\partial^2 C}{\partial S^2}$)
*   Replaces theoretical Greek formulas with purely explicit **Neural Gradients**.
*   Executes dual sequential PyTorch `autograd` loops (`create_graph=True`) chronologically to derive the exact mathematically true Gamma variance tracking Out-Of-The-Money distributions during "fat tails."

### 3. Frictional Reinforcement Learning (RL) Hedge Agent
The core pricing engine derives the analytically frictionless Neural Deltas. However, executing those Deltas across an actual exchange natively destroys capital through the **Bid/Ask Spread friction**.
*   A custom `gymnasium` environment fed with strictly empirical S&P 500 trajectories (including the Q1 2020 Black-Swan Crash).
*   An autonomous Artificial Intelligence (`Stable-Baselines3`) plays back 20-day options chunks, actively tracking a synthetic inventory portfolio. It elegantly learns the threshold between catastrophic tracking error and terminal slip destruction smoothly.

### 4. Direct Tier-1 Institutional Tick-Exchange Connectivity
*   Features a Live WebSocket orchestrator (`ib_insync`) tracking the `Interactive Brokers` Paper Trading API securely on Port 7497, reverting dynamically to `yfinance` limits upon structural failure.

## 🚀 Installation & Build Procedures

```bash
# Clone the Core Engine
git clone https://github.com/PiyushKBhattacharyya/Advanced-Derivatives-Pricing.git
cd Advanced-Derivatives-Pricing

# Mount the Virtual Environment
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

# Phase 1: Synthesize standard Neural Scales (Required structural metadata)
python run_pipeline.py

# Phase 2: Train the Deep BSDE Structural Weights
python src/train_deep_bsde.py

# Phase 3: Train the Frictional RL Pipeline against Empirical Data (50,000 steps)
python src/rl_agent.py

# Phase 4: Launch the Institutional Analytics Dashboard
streamlit run src/app.py
```

## 🔬 Scientific Validation
By tracking **Hedging portfolio deviations (Tracking Error P&L)**, this software graphically validates that during the COVID-19 crash, while legacy Black-Scholes hedges collapsed catastrophically off the underlying variance, the Deep Learning networks preserved physical capital tightly securely maintaining structural pricing boundaries.

## Live Link
https://deep-bdse-pricing.streamlit.app/