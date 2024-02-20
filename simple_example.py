'''
First example of bayesian estimation as suggested by the Gaussian book.
'''
import torch
import matplotlib.pyplot as plt

n_samples = 10
x_values = torch.linspace(-5, 5, n_samples)

y_values = torch.zeros(n_samples)
sigma_noise = 1.

for nth in range(n_samples):
	noise = torch.normal(0., sigma_noise, (1,))
	y_values[nth] = x_values[nth] * 2 + 0.4 + noise

plt.plot(x_values, y_values)
plt.grid()
plt.show()

########################################################################
####	Manipulate data to make them compatible with the model
#######################################################################
x_data = torch.ones(n_samples, 2)
for nth in range(n_samples):
	x_data[nth][1] = x_values[nth]


############################################################################
####	Using here the notation from the book in order to help readability
############################################################################
D = x_data.shape[1]		# Dimension of data
n = x_data.shape[0]		# Number of samples
X = x_data.T

# Prior covariance matrix
bigsigma = torch.eye(D)

i_bigsigma = torch.inverse(bigsigma)
A = torch.matmul(X, X.T) / (sigma_noise ** 2.)  + i_bigsigma
i_A = torch.inverse(A)

print(f"Dimension of A should be {D}x{D}, and is {A.shape}")

y = y_values.reshape(n, 1)
x_new = torch.tensor([1., 1.]).reshape(D, 1)

#tmp1 = torch.matmul(x_new.T, i_A)
#tmp2 = torch.matmul(tmp1, X)
#mean = torch.matmul(tmp2, y) / (sigma_noise ** 2.)

tmp1 = torch.matmul(i_A, X)
C = torch.matmul(tmp1, y) / (sigma_noise ** 2)


#####
n_pred = 20
x_to_pred = torch.ones(n_pred, 2)
x_to_pred[:, 1] = torch.linspace(-7, 7, n_pred)

y_predicted = torch.matmul(x_to_pred, C)

uncertainty = torch.zeros(n_pred)
for nth in range(n_pred):
	x_star = x_to_pred[nth].reshape(2, 1)
	uncertainty[nth] = torch.matmul(torch.matmul(x_star.T, i_A), x_star)
	uncertainty[nth] = torch.sqrt(uncertainty[nth])

plt.plot(x_to_pred[:, 1], y_predicted, linestyle = "dashdot", color = "orange")
plt.plot(x_to_pred[:, 1] + 2 * uncertainty, 
	y_predicted, linestyle = "dotted", color = "green")
plt.plot(x_to_pred[:, 1] - 2 * uncertainty, 
	y_predicted, linestyle = "dotted", color = "green")

plt.plot(x_values, y_values, color = "blue")
plt.grid()
plt.show()


