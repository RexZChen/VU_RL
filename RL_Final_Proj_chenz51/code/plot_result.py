"""
This script is for plotting the results of average validation returns of MAML in three different settings

Author: Zirong Chen
"""

import matplotlib
matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt

dir_name = "Res/157_average_return"
return_list_0 = np.loadtxt(dir_name + "/0_adapt.csv")
return_list_1 = np.loadtxt(dir_name + "/1_adapt.csv")
return_list_2 = np.loadtxt(dir_name + "/2_adapt.csv")

y = np.arange(0, return_list_1.shape[0])  # 150
plt.figure(figsize=(30, 20))

plt.plot(y, return_list_0, label="0 adapt", linewidth=3.0, ms=10)
plt.plot(y, return_list_1, label="1 adapt", linewidth=3.0, ms=10)
plt.plot(y, return_list_2, label="2 adapt", linewidth=3.0, ms=10)

plt.legend(prop={'size': 30})

plt.xlabel("Number of iterations", fontsize=30)
plt.ylabel("Average Validation Return", fontsize=30)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)

plt.savefig("result.png")

