import matplotlib.pyplot as plt
import numpy as np
plt.figure(figsize=(20,20))


data = np.load("data.npy", allow_pickle=True)
# print(data)
plt.plot(data[0], data[1], alpha=0.6, color="red")
plt.show()
