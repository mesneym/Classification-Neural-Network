import matplotlib.pyplot as plt
import numpy as np


fig,ax = plt.subplots()
fig,ax2 = plt.subplots()

epochs = np.arange(20)

loss = [0.213,0.188,0.157,0.1336,0.113,0.0772,0.07334, 0.115,0.094,0.1094,0.0658,0.05031,0.0430,0.0245,0.0282,0.0794,0.0455,0.0345,0.0478,0.0571]
acc = [0.665,0.761,0.794,0.825,0.822,0.85,0.866,0.848,0.882,0.89,0.894,0.899,0.903,0.912,0.917,0.88,0.896,0.945,0.932,0.912]

acc =np.array(acc) * 100
loss = np.array(loss)

ax.plot(epochs,acc)
ax.set_title("Accuracy Vs Epochs")
ax.set_ylabel("Accuracy")
ax.set_xlabel("Epochs")

ax2.plot(epochs,loss)
ax2.set_title("loss Vs Epochs")
ax2.set_ylabel("Loss")
ax2.set_xlabel("Epochs")
plt.show()




