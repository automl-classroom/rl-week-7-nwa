import pandas as pd
import matplotlib.pyplot as plt

rnd = pd.read_csv("rl_exercises/week_7/training_data_seed_0.csv")          
dqn = pd.read_csv("rl_exercises/week_7/training_data_dqn_seed.csv")         

rnd["smoothed"] = rnd["rewards"].rolling(window=10).mean()
dqn["smoothed"] = dqn["rewards"].rolling(window=10).mean()

plt.figure(figsize=(10, 5))
plt.plot(rnd["steps"], rnd["smoothed"], label="RND-DQN")
plt.plot(dqn["steps"], dqn["smoothed"], label="Epsilon-Greedy DQN")
plt.xlabel("Environment Steps")
plt.ylabel("Episode Reward (Smoothed)")
plt.title("Comparison: RND vs. Epsilon-Greedy DQN")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("comparison_plot.png") 
plt.show()
