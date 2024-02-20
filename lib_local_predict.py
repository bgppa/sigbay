'''
This script contains supporting functions specifically designed for the
LOCAL prediction script.
'''
import torch
import torch.nn as nn
import torch.optim as optim
from swutils import my_signature, isdecreasing
from swutils import add_zero_1d as add_zero
from swutils import augment_1d as simple_augment
import matplotlib.pyplot as plt

def predict_from (time_chunk, model, sig_depth, len_future, scaler):
	'''
	Starting from a single time chunk, predict multiple values for
	the future by an interative model evaluation.
	'''
	future_values = torch.zeros(len_future)
	# curr_chunk contains the PURE, ORIGINAL CHUNKS, not in log	
	curr_chunk = time_chunk.clone()
	for nth in range(len_future):
		start_val = curr_chunk[0]
		# take the log_return of the current chunk
		tmp1 = torch.log(curr_chunk / start_val) * scaler
		tmp2 = simple_augment(tmp1).unsqueeze(0)
		tmp3 = my_signature(tmp2, sig_depth)

		log_predicted = model(tmp3).detach()
		predicted = torch.exp(log_predicted / scaler) * start_val

		future_values[nth] = predicted.item()
		# shift the current chunk and add the new value
		tmp4 = curr_chunk[1:].clone()
		curr_chunk[:-1] = tmp4
		curr_chunk[-1] = predicted

	# Now I have an array containing len_future future values
	# predicted from the model starting from time_chunk
	return future_values
#---


def reliability_analysis (all_chunks, true_vals, model, sig_depth,
	len_future, scaler):
	'''
	Perform multiple long term predictions and compare the true with
	the predicted values, by taking their difference with the curren ones
	and evaluating if their sign match.
	So here I am not evaluating the quantitative error, but I am rather
	focusing on a qualitative analysis of trends.
	'''
	assert(all_chunks.shape[0] == true_vals.shape[0])
	n_points = all_chunks.shape[0] - len_future + 1
	truedata_n_increase = 0.

	# Precision matrix to store how many trends I guessed
	precision_matrix = torch.zeros((2,2))
	for nth in range(n_points):
		# Store the current value
		curr_value = all_chunks[nth][-1]
		# True future value
		true_future = true_vals[nth + len_future - 1]
		# Compute the _predicted_ future value	
		all_predicted = predict_from(all_chunks[nth], model,
					sig_depth, len_future, scaler)
		pred_future = all_predicted[-1]
		# Compare their signs and store the result in the class matrix 
		delta_true = true_future - curr_value
		# Keep track of how many times the increase happens in real
		if delta_true >= 0:
			truedata_n_increase += 1
		delta_pred = pred_future - curr_value
		if (delta_true >= 0) and (delta_pred >= 0):
			precision_matrix[1][1] += 1
		if (delta_true >= 0) and (delta_pred < 0):
			precision_matrix[1][0] += 1
		if (delta_true < 0) and (delta_pred >= 0):
			precision_matrix[0][1] += 1
		if (delta_true < 0) and (delta_pred < 0):
			precision_matrix[0][0] += 1

	# Convert the matrix into a percentage form
	precision_matrix = (precision_matrix / n_points * 100)
	truedata_increase = truedata_n_increase / n_points * 100
	return precision_matrix, truedata_increase
#---

def imprecision(all_chunks, true_vals, model, sig_depth, ln_future):
	'''
	Possible candidate for my loss function?
	To minimize the imprecision?
	'''
	m = reliability_analysis(all_chunks,true_vals,model,sig_depth,ln_future)
	# adding 1e-3 just to avoid possible zero values and then nan
	result = 1. / (m[0][0] + m[1][1] + 1e-3)
	return result
#---


def trend_training(dim, e_units, x_chunks_train, y_vanilla_train,
			x_chunks_val, y_vanilla_val, lr, strength = 5):

#def trend_training(dim, e_units, x_train, y_train, x_val, y_val, lr):
	'''
	This is a classic training routine for Pytorch, in our case we are fine
	with a simple linear model, mseloss and a learning rate as parameter.
	'''
	model = nn.Linear(dim, 1, bias = False)
	optimizer = optim.Adam(model.parameters(), lr=lr)

#	loss_fn = nn.MSELoss()
	# I will check that the losses are following a decreasing behavior
	monitor_size = 5
	last_losses = torch.zeros(monitor_size)
	# Monitoring the loss evolution during training
	n_epochs = 10_000 * e_units
	hist_loss_train = torch.ones(n_epochs)
	hist_loss_val = torch.ones(n_epochs)
	# Classic training loop
	for nth in range(n_epochs):
		optimizer.zero_grad()

		loss_train = imprecision(x_chunks_train, y_vanilla_train,
			model, dim, strength)

		with torch.no_grad():
			loss_val = imprecision(x_chunks_val, y_vanilla_val,
				model, dim, strength)
			if (nth % 1000 == 0):
				print(f"{nth+1}/{n_epochs}")
				print(f"t: {loss_train.item():.3e} ", end=' ')
				print(f"v:{loss_val.item():.3e}")
				hist_loss_train[nth] = loss_train.item()
				hist_loss_val[nth] = loss_val.item()
				if nth < monitor_size:
					last_losses[nth] = loss_val.item()
				else:
					tmp = torch.zeros(monitor_size)
					tmp[:-1] = last_losses[1:]
					tmp[-1] = loss_val.item()
					last_losses = tmp
				if (nth > 0) and (nth % 1000 == 0):
					if not isdecreasing(last_losses):
						print("Last monitored losses: ")
						print(last_losses)
						ch = input("Stop straining?")
						if len(ch) > 0:
							ch = ch.upper()[0]
							if ch == 'Y':
								break
		loss_train.backward()
		optimizer.step()
	# endfor
	print(f"Training ended!")
	plt.plot(hist_loss_train[1000:nth], label='train')
	plt.plot(hist_loss_val[1000:nth], label='val')
	plt.grid()
	plt.legend()
	plt.title("Loss function evolution")
	plt.show()
	coefficients = list(model.parameters())[0][0].detach()
	plt.plot(range(1, dim+1), coefficients)
	plt.axhline(y=0, color="black", linestyle="dashdot")
	plt.grid()
	plt.title(f"The {dim} parameters of the linear model")
	plt.show()
	return model
#---

