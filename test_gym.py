from gym import spaces
import numpy as np

try:
    print("Testing MultiBinary(0)...")
    s = spaces.MultiBinary(0)
    print("Success")
except Exception as e:
    print(f"Failed: {e}")

try:
    print("Testing MultiDiscrete([])...")
    s = spaces.MultiDiscrete([])
    print("Success")
except Exception as e:
    print(f"Failed: {e}")

try:
    print("Testing MultiBinary([0])...")
    s = spaces.MultiBinary([0])
    print("Success")
except Exception as e:
    print(f"Failed: {e}")
