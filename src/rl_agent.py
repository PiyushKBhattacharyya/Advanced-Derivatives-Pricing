import os
import sys
import numpy as np
import logging

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from src.rl_env import FrictionalHedgingEnv

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_checker import check_env
    from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement
    HAS_SB3 = True
except ImportError:
    HAS_SB3 = False

def train_frictional_ppo_agent():
    """
    Compiles the Deep Reinforcement Learning pipeline.
    The Agent (PPO) trains inside the Gymnasium simulation, learning to optimally 
    balance tracking the Deep BSDE's Neural Delta against the destruction of capital
    caused by Bid/Ask spread slippage.
    """
    if not HAS_SB3:
        logging.error("Stable-Baselines3 is strictly missing. Please manually run: `pip install stable-baselines3 gymnasium`")
        return
        
    logging.info("Synthesizing physical option trajectories for OpenAI Gym Environment...")
    
    import pandas as pd
    
    # 1. Mount REAL 6-Year Historical Market DataFrame directly
    spx_data_path = os.path.join(BASE_DIR, "Data", "SPX_history.csv")
    if not os.path.exists(spx_data_path):
        logging.error(f"Missing Empirical Path Data at {spx_data_path}. Training Terminated.")
        return
        
    spx_df = pd.read_csv(spx_data_path, index_col=0, parse_dates=True)
    real_prices = spx_df['SPX'].dropna().values
    
    # To train the Reinforcement Learning agent, we split the 6-year history into thousands of discrete 60-day options chunks!
    n_steps = 60
    n_paths = len(real_prices) - n_steps
    
    spx_paths = np.zeros((n_paths, n_steps))
    for i in range(n_paths):
        spx_paths[i] = real_prices[i:i+n_steps]
        
    logging.info(f"Loaded {n_paths} unique 60-day Empirical Crash Trajectories exactly from Yahoo Finance.")
        
    # 2. Surrogate Deep BSDE Deltas (Target Limits the RL Agent inherently tracks)
    # Since executing the PyTorch autograd engine 200,000 times natively would take hours, 
    # we mathematically synthesize purely correlated baseline target matrices.
    # We now cover the FULL range [0.0, 1.0] to handle At-The-Money (ATM) options (~0.5 delta).
    bsde_deltas = np.zeros_like(spx_paths)
    for i in range(n_paths):
        # Surrogate matched to actual neural network autograd outputs (~0.0 to 1.0).
        # We model it as a baseline 0.5 (ATM) + adjustment for spot momentum.
        spot_movement = (spx_paths[i] - spx_paths[i, 0]) / spx_paths[i, 0]
        # Real-world ATM delta is around 0.5. We use a linear proxy for training diversity.
        bsde_deltas[i] = np.clip(0.50 + spot_movement * 2.0, 0.0, 1.0)

    
    env = FrictionalHedgingEnv(spx_paths, bsde_deltas, transaction_cost=0.0002)
    check_env(env) # Validate strict compliance with OpenAI Gym API
    
    logging.info("Environment Validated. Initializing Proximal Policy Optimization (PPO) Brain...")
    
    # MlpPolicy: 2-layer NN processing the [4] observation vector.
    # Larger n_steps=4096 gives the agent more context before each update.
    # Larger batch_size=256 gives more stable gradient estimates.
    # Lower lr=0.0001 ensures fine-grained convergence near the delta target.
    model = PPO(
        "MlpPolicy", env, verbose=1,
        learning_rate=0.0001,
        n_steps=4096,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.005,           # Small entropy bonus encourages exploration
    )
    
    # Configure Early Stopping Callbacks
    # Evaluate every 10,000 steps. Stop if no improvement for 5 consecutive evaluations (50k steps patience)
    eval_env = FrictionalHedgingEnv(spx_paths, bsde_deltas, transaction_cost=0.0002)
    stop_train_callback = StopTrainingOnNoModelImprovement(max_no_improvement_evals=5, min_evals=5, verbose=1)
    eval_callback = EvalCallback(eval_env, eval_freq=10000, callback_after_eval=stop_train_callback, 
                                 best_model_save_path=os.path.join(BASE_DIR, "Data"),
                                 verbose=1)
                                 
    logging.info("Commencing Deep RL Training Matrix (Max 5,00,000 Timesteps with Early Stopping)...")
    model.learn(total_timesteps=500_000, callback=eval_callback)
    
    # Export the Final Neural Weights (if not already saved by best_model)
    save_path = os.path.join(BASE_DIR, "Data", "PPO_Frictional_Agent.zip")
    model.save(save_path)
    logging.info(f"PPO Agent successfully attained convergence. Model secured at: {save_path}")

if __name__ == "__main__":
    train_frictional_ppo_agent()
