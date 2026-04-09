import gym
from pyhailing.ridehail_env import RidehailEnv
from pyhailing.core import Jobs
import numpy as np

def verify():
    # 1. Initialize environment
    config = RidehailEnv.DIMACS_CONFIGS.SUI.copy()
    config["nickname"] = "verify"
    env = RidehailEnv(**config)
    obs = env.reset()
    
    print("--- 1. Observation Test ---")
    if "v_soc" in obs:
        print(f"SUCCESS: 'v_soc' found in observation. First 5 vehicles SOC: {obs['v_soc'][:5]}")
    else:
        print("FAILURE: 'v_soc' not found in observation.")
        return

    print("\n--- 2. Energy Consumption & Charging Test ---")
    # Force an action to see movement
    action = env.get_random_action()
    initial_soc = env._vehicles["soc"].copy()
    
    # Step the environment
    obs, reward, done, info = env.step(action)
    new_soc = env._vehicles["soc"]
    
    # Check for changes
    diffs = new_soc - initial_soc
    moving = (env._vehicles["j1m"] != Jobs.IDLE)
    charging = (env._vehicles["j1m"] == Jobs.IDLE)
    
    print(f"Mean SOC Change: {np.mean(diffs):.4f} kWh")
    if any(diffs[moving] < 0):
        print("SUCCESS: Moving vehicles consumed energy.")
    if any(diffs[charging] > 0):
        print("SUCCESS: Idle vehicles gained energy (charged).")

    print("\n--- 3. Strict Feasibility Test ---")
    # Drain one vehicle's SOC manually to just above reserve (5 kWh)
    vehicle_idx = 0
    env._vehicles.at[vehicle_idx, "soc"] = 6.0 # Barely above 5kWh reserve
    
    print(f"Vehicle 0 SOC forced to: {env._vehicles.at[0, 'soc']} kWh")
    
    # Try to assign a long trip to vehicle 0
    # Any trip usually requires > 1kWh (6.6km)
    # Plus dest-to-charger which is at least 0.1km
    
    # Get a request
    pending_reqs = env._get_pending_requests()
    if len(pending_reqs) > 0:
        vs = env._vehicles.iloc[[0]]
        reqs = pending_reqs.iloc[[0]]
        feasible = env._check_assignment_feasibility(vs, reqs)
        print(f"Feasibility for Vehicle 0 on Request 0: {feasible[0]}")
        
        # Now drop SOC below reserve
        env._vehicles.at[vehicle_idx, "soc"] = 4.0
        feasible_low = env._check_assignment_feasibility(vs, reqs)
        print(f"Feasibility for Vehicle 0 (SOC=4.0) on Request 0: {feasible_low[0]}")
        if not feasible_low[0]:
            print("SUCCESS: Low SOC vehicle correctly marked as infeasible.")

if __name__ == "__main__":
    verify()
