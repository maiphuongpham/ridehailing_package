import gym
from pyhailing.ridehail_env import RidehailEnv
from pyhailing.core import Jobs
import numpy as np

def paper_random_agent(env, obs):
    """
    Implements a random agent that matches the paper's description:
    1. Zero random rejections (accepts all requests).
    2. Assigns requests to eligible vehicles at random.
    """
    # Start with a random action from the environment's own method
    # This ensures all vehicles needing instructions get them.
    action = env.get_random_action()
    
    # 1. Zero random rejections (override).
    # The paper's Random agent doesn't randomly throw away revenue.
    action["req_rejections"][:] = 0
    
    # We'll leave the random assignments and repositioning as generated 
    # by env.get_random_action(), as it already picks from eligible 
    # samples according to the action space rules.
    
    return action

def evaluate(config, num_episodes=5):
    env = RidehailEnv(**config)
    total_rewards = []
    
    print(f"\nEvaluating Paper-style Random agent over {num_episodes} episodes...")
    for ep in range(num_episodes):
        obs = env.reset()
        done = False
        ep_reward = 0
        steps = 0
        
        while not done:
            action = paper_random_agent(env, obs)
            obs, reward, done, info = env.step(action)
            ep_reward += reward
            steps += 1
            
        total_rewards.append(ep_reward)
        print(f"Episode {ep+1}: Reward = {ep_reward:.2f}, Steps = {steps}")
        
    mean_reward = np.mean(total_rewards)
    std_reward = np.std(total_rewards)
    print(f"\nResult for Paper-style Random: Mean Reward = {mean_reward:.2f} +/- {std_reward:.2f}")
    return mean_reward

if __name__ == "__main__":
    # Access SUI configuration as an attribute
    config = RidehailEnv.DIMACS_CONFIGS.SUI.copy()
    config["nickname"] = "researcher_refined"
    
    mean_reward = evaluate(config, num_episodes=5)
    
    print("\nComparison with Figure 6:")
    print(f"Paper Random:  $6,815")
    print(f"My Reproduction: ${mean_reward:.2f}")
