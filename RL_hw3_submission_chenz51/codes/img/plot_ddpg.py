import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


df = pd.read_csv("DDPG_LunarLander")

print(df)
#
traj = df["traj"]
pi_loss = df["pi_loss"]
q_loss = df["q_loss"]
avg_return = df["avg_returns"]
returns = df["avg_returns"]
lens = df["length"]
#
plt.plot(traj, q_loss, color="blue")
plt.savefig("DDPG_LunarLander_q_loss", dpi=500)
plt.show()

# avg_return = np.array(avg_return)
# avg_return = avg_return.reshape(-1, 100)
#
# _len = np.array(lens)
# _len = _len.reshape(-1, 100)
#
# avg_list = []
# traj_list = []
# len_list = []
# for idx in range(len(avg_return)):
#     avg = np.mean(avg_return[idx])
#     avg_len = np.mean(_len[idx])
#     avg_list.append(avg)
#     len_list.append(avg_len)
#     traj_list.append((idx + 0.5) * 100)
#
# # print(avg_list)
#
# plt.plot(traj, returns, color="blue")
# plt.plot(traj_list, avg_list, color="red")
# # plt.plot(traj_list, len_list, color="yellow")
# plt.savefig("DDPG_LunarLander_w_avg", dpi=500)
# plt.show()
