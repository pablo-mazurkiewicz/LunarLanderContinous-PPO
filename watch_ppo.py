import gymnasium as gym
import cv2
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv

ENV_ID = "LunarLanderContinuous-v3"
SEED = 0 
MODEL_PATH = f"ppo_lunarlander_seed{SEED}"
STATS_PATH = f"vec_normalize_seed{SEED}.pkl"
#^ if want to see model for trained on windy enviroment - add _WINDY before _seed in both paths


def main():
    print(f"Ładowanie modelu: {MODEL_PATH}")
    print(f"Ładowanie statystyk: {STATS_PATH}")

    env = DummyVecEnv([lambda: gym.make(ENV_ID, render_mode="rgb_array", enable_wind=True, wind_power=15.0, turbulence_power=1.5)])
#Remove those if you want to see the results for perfect conditions: enable_wind=True, wind_power=15.0, turbulence_power=1.5
#tensorboard --logdir=C:/rl_logs - to see the tensorboard graphs
        
    try:
        env = VecNormalize.load(STATS_PATH, env)
    except FileNotFoundError:
        print("Nie znaleziono pliku .pkl sprawdz czy trening sie zapisał.")
        return

    env.training = False
    env.norm_reward = False

    model = PPO.load(MODEL_PATH, env=env)

    obs = env.reset()
    
    total_reward = 0.0
    step = 0
    episode = 1

    while True:
        action, _ = model.predict(obs, deterministic=True)
        
        obs, rewards, dones, infos = env.step(action)

        reward_scalar = rewards[0]
        done_scalar = dones[0]

        total_reward += reward_scalar
        step += 1

        frame = env.envs[0].render()
        
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        cv2.putText(frame, f"Epizod: {episode}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Krok: {step}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Nagroda: {total_reward:.1f}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("PPO LunarLander (Normalized)", frame)

        if cv2.waitKey(20) & 0xFF == 27:
            break

        if done_scalar:
            print(f"Epizod {episode} zakończony. Wynik: {total_reward:.2f}")
            obs = env.reset()
            total_reward = 0.0
            step = 0
            episode += 1

    cv2.destroyAllWindows()
    env.close()

if __name__ == "__main__":
    main()