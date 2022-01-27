import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


df = pd.read_csv("VPG_LunarLander")

print(df)

epoch = df["Epochs"]
vf_loss = df["pi_loss"]
q_loss = df["vf_loss"]
avg_return = df["avg_returns"]
returns = df["avg_returns"]

plt.plot(epoch, q_loss, color="blue")
plt.savefig("VPG_LunarLander_vf_loss", dpi=500)
plt.show()

avg_return = np.array(avg_return)
avg_return = avg_return.reshape(-1, 100)

avg_list = []
epo_list = []
for idx in range(len(avg_return)):
    avg = np.mean(avg_return[idx])
    avg_list.append(avg)
    epo_list.append((idx + 0.5) * 100)

# print(avg_list)
#
# plt.plot(epoch, returns, color="blue")
# plt.plot(epo_list, avg_list, color="red")
# plt.savefig("VPG_LunarLander_w_avg", dpi=500)
# plt.show()
