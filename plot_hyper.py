'''
In this script we basically simply plot the obtained
hyperparameters
'''
import torch
import matplotlib.pyplot as plt

f = open("hypermapameters.txt", "r")

windows = []
accs = []
spreads =[]

for line in f:
	tmp = line.strip().split("|")
	windows.append(int(tmp[1]))
	accs.append(float(tmp[2]))
	spreads.append(float(tmp[3]))

dataname = tmp[0]

n_observations = len(windows)

for nth in range(n_observations):
	plt.scatter(accs[nth], spreads[nth], color = "orange")
	plt.text(accs[nth], spreads[nth], f"{nth}")
plt.axvline(x = 80, color = "green", linestyle = "dashdot")
plt.axhline(y = 3, color = "green", linestyle = "dashdot")
plt.xlabel("accuracy [to maximize]")
plt.ylabel("spread [to minimize]")
plt.grid()
plt.title(f"Window search for {dataname}")
plt.show()

chosen = int(input("Insert che chosen point: "))
if chosen >= 0 and chosen < n_observations:
	print(f"Point number {chosen} has:")
	print(f"Accuracy {accs[chosen]}, Spread {spreads[chosen]}")
	print(f"It corresponds to a WINDOW OF {windows[chosen]}")
