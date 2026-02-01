import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv

ENV_ID = "LunarLanderContinuous-v3"
SEED = 0
MODEL_PATH = f"ppo_lunarlander_seed{SEED}"
STATS_PATH = f"vec_normalize_seed{SEED}.pkl"
N_EVAL_EPISODES = 200

def evaluate_windy_conditions():
    print(f"--- TEST lądowania ---")


    env_kwargs = {
        #variables for test in windy conditions
        #"enable_wind": True,
        #"wind_power": 15.0,
        #"turbulence_power": 1.5 
    }
    
    env = DummyVecEnv([lambda: gym.make(ENV_ID, **env_kwargs)])

    try:
        env = VecNormalize.load(STATS_PATH, env)
    except FileNotFoundError:
        print("Brak pliku normalizacji!")
        return

    env.training = False
    env.norm_reward = False

    model = PPO.load(MODEL_PATH, env=env)

    success_count = 0
    all_rewards = []

    print("Rozpoczynam procedure lądowania...")
    
    for i in range(N_EVAL_EPISODES):
        obs = env.reset()
        done = False
        total_reward = 0.0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = env.step(action)
            total_reward += rewards[0]
            done = dones[0]

        all_rewards.append(total_reward)
        
        if total_reward >= 200:
            success_count += 1

    mean_reward = np.mean(all_rewards)
    std_reward = np.std(all_rewards)
    success_rate = (success_count / N_EVAL_EPISODES) * 100

    print("\n" + "="*40)
    print(f"WYNIKI LĄDOWANIA")
    print("="*40)
    print(f"Średnia nagroda:      {mean_reward:.2f}")
    print(f"Odchylenie standard.: {std_reward:.2f}")
    print(f"Liczba sukcesów:      {success_count}/{N_EVAL_EPISODES}")
    print(f"SKUTECZNOŚĆ:          {success_rate:.1f}%")
    print("="*40)

    env.close()

if __name__ == "__main__":
    evaluate_windy_conditions()