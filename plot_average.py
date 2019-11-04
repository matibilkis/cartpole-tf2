import matplotlib.pyplot as plt
import numpy as np
plt.figure(figsize=(20,20))
import os
os.chdir("datas")
data=np.zeros(100)
for agent in range(27):
    data = data + np.load("data"+str(agent)+".npy", allow_pickle=True)[1]
data = data/27
plt.plot(np.arange(100), data, alpha=0.6, color="red")
plt.xlabel("Episodes")
plt.ylabel("Times unitl pole falls")
# plt.show()
os.chdir("..")
plt.savefig("average_27_agents.png")
# print(data)
# d = np.load("data17.npy",allow_pickle=True)
