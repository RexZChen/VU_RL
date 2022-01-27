import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df = pd.read_csv("PPO_BipedalWalker_256_20000")

print(df)

epoch = df["epoch"]
pi_loss = df["pi_loss"]
vf_loss = df["vf_loss"]
avg_rewards = df["avg_reward"]
avg_steps = df["avg_steps"]

avg_return = np.array(avg_rewards)
avg_return = avg_return.reshape(-1, 100)

avg_list = []
epo_list = []
for idx in range(len(avg_return)):
    avg = np.mean(avg_return[idx])
    avg_list.append(avg)
    epo_list.append((idx + 0.5) * 100)


plt.plot(epoch, avg_rewards, color="blue")
plt.plot(epo_list, avg_list, color="red")
plt.savefig("PPO_BipedalWalker_over", dpi=500)
plt.show()

# plt.show()
# plt.plot(epoch, avg_steps, color="green")
