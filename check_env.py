import gymnasium as gym

ENV_ID = "LunarLanderContinuous-v3"

def main():
    env = gym.make(ENV_ID)
    obs, info = env.reset(seed=0)
    print("Obs shape:", obs.shape)
    print("Action space:", env.action_space)
    print("Observation space:", env.observation_space)

    returns = []
    for episode in range(5):
        obs, _ = env.reset()
        total_r = 0.0
        for _ in range(200):
            action = env.action_space.sample()
            obs, r, terminated, truncated, info = env.step(action)
            total_r += r
            if terminated or truncated:
                break
        returns.append(total_r)
    
    env.close()
    print(f"Random policy returns: {returns}")
    print(f"Mean: {sum(returns) / len(returns):.2f}")

if __name__ == "__main__":
    main()
