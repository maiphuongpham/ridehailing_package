import time
import numpy as np
from pyhailing import RidehailEnv

def evaluate(env_config, agent_type, num_episodes=5):
    env = RidehailEnv(**env_config)
    all_rewards = []
    
    print(f"\nEvaluating {agent_type} agent over {num_episodes} episodes...")
    
    for ep in range(num_episodes):
        obs = env.reset()
        done = False
        total_reward = 0
        steps = 0
        start_time = time.time()
        
        while not done:
            if agent_type == "noop":
                action = env.get_noop_action()
            elif agent_type == "random":
                action = env.get_random_action()
            else:
                raise ValueError("Unknown agent type")
                
            obs, reward, done, info = env.step(action)
            total_reward += reward
            steps += 1
            
        duration = time.time() - start_time
        all_rewards.append(total_reward)
        print(f"Episode {ep+1}: Reward = {total_reward:.2f}, Steps = {steps}, Duration = {duration:.2f}s")
        
    mean_reward = np.mean(all_rewards)
    std_reward = np.std(all_rewards)
    print(f"Result for {agent_type}: Mean Reward = {mean_reward:.2f} +/- {std_reward:.2f}")
    return mean_reward

if __name__ == "__main__":
    config = RidehailEnv.DIMACS_CONFIGS.SUI
    config["nickname"] = "researcher"
    
    # Evaluate No-op
    noop_reward = evaluate(config, "noop", num_episodes=2)
    
    # Evaluate Random
    random_reward = evaluate(config, "random", num_episodes=2)
    
    print("\nComparison with paper (Primary 1,400/14 Case):")
    print(f"Paper Drafter: ~$10,898")
    print(f"Paper Reopt:   ~$9,067")
    print(f"Paper Random:  (Visualized in Fig 5, likely negative or near zero)")
