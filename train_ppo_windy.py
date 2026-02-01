"""
PPO Training Script - WINDY VERSION
Saves models with '_WINDY' suffix to preserve original non-wind models.
"""
import sys
import json
import os
import shutil
import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure

ENV_ID = "LunarLanderContinuous-v3"
SEEDS = [0, 42, 123, 999, 2025]
TOTAL_TIMESTEPS = 2_000_000 
SAFE_LOG_DIR = "C:/rl_logs/ppo_windy" 

class LogCallback(BaseCallback):
    """Logs episode returns during training"""
    def __init__(self, log_dir="ppo_logs_windy", seed=0):
        super().__init__()
        self.log_dir = log_dir
        self.seed = seed
        self.returns = []
        self.current_return = 0.0
        os.makedirs(log_dir, exist_ok=True)
    
    def _on_step(self) -> bool:
        if "dones" not in self.locals or "rewards" not in self.locals:
            return True
        
        dones = self.locals["dones"]
        rewards = self.locals["rewards"]
        
        for i, (done, reward) in enumerate(zip(dones, rewards)):
            self.current_return += reward
            if done:
                self.returns.append(float(self.current_return))
                self.current_return = 0.0
        return True
    
    def _on_training_end(self) -> None:
        log_file = os.path.join(self.log_dir, f"returns_windy_seed{self.seed}.json")
        with open(log_file, "w") as f:
            json.dump(self.returns, f, indent=2)
        print(f"Saved training log: {log_file}")


def train_ppo_windy(seed=0):
    print(f"Training PPO (WINDY) with seed={seed}")
    
    n_envs = 8
    env_kwargs = {'enable_wind': True, 'wind_power': 15.0, 'turbulence_power': 1.5}
    
    env = make_vec_env(ENV_ID, n_envs=n_envs, seed=seed, env_kwargs=env_kwargs)
    
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)

    model = PPO(
        policy="MlpPolicy",
        env=env,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        learning_rate=2.5e-4, 
        clip_range=0.2,
        ent_coef=0.0,
        seed=seed,
        verbose=1,
        device="cpu"
    )
    
    logger_path = os.path.join(SAFE_LOG_DIR, f"seed_{seed}")
    print(f"--- LOGI TENSORBOARD BĘDĄ W: {logger_path} ---")
    
    new_logger = configure(logger_path, ["stdout", "tensorboard"])
    model.set_logger(new_logger)
    
    callback = LogCallback(log_dir="ppo_logs_windy", seed=seed)
    
    try:
        model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callback)
    except Exception as e:
        print(f"big error: {e}")
        raise e
    finally:
        model_name = f"ppo_lunarlander_WINDY_seed{seed}"
        stats_name = f"vec_normalize_WINDY_seed{seed}.pkl"
        
        model.save(model_name)
        env.save(stats_name)
        print(f"Zapisano model WINDY: {model_name}")
        print(f"Zapisano statystyki WINDY: {stats_name}")
        env.close()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        try:
            arg = int(sys.argv[1])
            seed = SEEDS[arg] if arg < len(SEEDS) else arg
        except ValueError:
            seed = 0
    else:
        seed = 0
    
    train_ppo_windy(seed=seed)