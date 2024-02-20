'''
Full predictor script using a LOCAL log return transform.
Please specify in the argument:
	(1) the datafile to read
	(2) the length of the time windowing
'''
import torch
import matplotlib.pyplot as plt
import sys
import os
import torch.optim as optim
import torch.nn as nn

# Supporting functions from the local libraries, in this folder
from swutils import logback
#from swutils import strict_positive as ispositive
#from swutils import my_signature, diagonal_test
#from libgbm import expected_path
from lib_local_predict import predict_from, reliability_analysis
from libtsm import augment, get_1var
from libsig import truncation_err, plot_signature, my_signature
from libmdl import linear_training, diagonal_test
torch.manual_seed(0)

TESTDATA = False # if True, discard part of the final series to use as test
FUTURE = 0	# number future points to predict wrt val data size
IGNORE = 0
GIVENAVG = 0	# True if the average path is provided (gbm/stoch proc case)
DEBUG = 0
RESCALER = 2.

sig_depth = 3

def msg():
	if DEBUG:
		input("DEBUG - Press ENTER to continue")
#---


# Check if the correct arguments are given
if (len(sys.argv) == 3):
	filename = sys.argv[1]
	window = int(sys.argv[2])
	if (os.path.exists(filename)):
		if (window <= 1):
			print(f"Windowing {window} not valid. Must be > 1.")
			quit()
	else:
	        print(f"File {filename} does not exist!")
	        quit()
else:
	print(f"Invalid syntax. Please use:")
	print(f"{sys.argv[0]} filename windowing")
	quit()


################################################################
#	Script is successfully initialized
################################################################
print(f"Opening {filename}; windowing of {window} steps")
#time_series = torch.load(filename).reshape(-1,)[:-IGNORE]
time_series = torch.load(filename).reshape(-1,)
if IGNORE > 0:
	time_series = time_series[:-IGNORE]
#if (is_spositive(time_series) == 0):
if (False in (time_series > 0)):
	print(f"FATAL: since this script works with logreturn,")
	print(f" the data are required to be strictly positive.")
	quit()

# Read the original series, backup it ("og") and determine how far we predict
og_time_series = time_series
n_times = time_series.shape[0]
len_future = FUTURE

# reserve part of the original series for blind test data	
if TESTDATA:
	y_test = time_series[-len_future:]
	# Update the time series to consider the shorter one
	time_series = time_series[:-len_future]
	n_times = time_series.shape[0]
	plt.plot(range(n_times, n_times + len_future), y_test, color="orange",
		label = "test data")

# Plot the read time series
plt.plot(time_series)
plt.axvline(x=window, color="red", linestyle="dashed")
plt.legend()
plt.grid()
my_title = f"Time Series from {filename} "
if TESTDATA:
	my_title = my_title + f"({n_times+len_future} nodes)"
else:
	my_title = my_title + f"({n_times} nodes)"
plt.title(my_title)
plt.show()


###############################################################
##	Data windowing
##############################################################
n_samples = n_times - window
if (n_samples < 40):
    print(f"The number of samples after windowing is {n_samples}. Too low.")
    quit()

starting_values = torch.zeros(n_samples)	# Starting values of every 
rescalers = torch.zeros(n_samples)
x_win = torch.zeros(n_samples, window)	# windowed data
y_tmp_full = torch.zeros(n_samples, 1)	# corresponding next values, log
y_vanilla = torch.zeros(n_samples, 1)	# corresponding next values, untouched

for nth in range(n_samples):
	x_win[nth] = time_series[nth : nth + window]
	starting_values[nth] = x_win[nth][0]
	y_vanilla[nth] = time_series[nth + window]
	tmp = time_series[nth + window] / starting_values[nth]
	y_tmp_full[nth][0] = torch.log(tmp)

# The last chunk does not have a y value, since time ends here
# Indeed, we want to predict its next value
last_chunk = time_series[nth + 1 : nth + window + 1]

print(f"Full time series: {time_series}")
print(f"Windowed ORIGINAL time series:")
print(f"{x_win}")
print(f"log y (NOT RESCALED) is: {y_tmp_full}")
print(f"Last chunk (excluded from database): {last_chunk}")
msg()


################################################################
##	Data preprocessing via the signature transform
##	Want to rescale them so to uniform the error?
################################################################
# Compute the log returns of the dataset
x_tmp_logret = torch.zeros(n_samples, window)
for nth in range(n_samples):
	x_tmp_logret[nth] = torch.log(x_win[nth] / starting_values[nth])

# Rescale the whole dataset so to better control the signature error
final_rescaler = RESCALER / get_1var(x_tmp_logret.unsqueeze(2)).mean()
x_logret = x_tmp_logret * final_rescaler
y_full = y_tmp_full * final_rescaler
# Conclude the dataset by augmenting the logreturns
x_aug = augment(x_logret.unsqueeze(2))
#
mean_length = get_1var(x_aug).mean()
print(f"Complete dataset of mean length {mean_length : .3f}")
print(f"{truncation_err(sig_depth, mean_length):.2f}")
input("OK?")

# Generating now the FULL dataset composed of signature elements
x_sig = my_signature(x_aug, sig_depth)
sig_len = x_sig.shape[1]

print(f"Ready with a database of shape {x_sig.shape}")
assert(n_samples == x_sig.shape[0])

# The first half of the dataset goes into training data, the second validation
# we add an offset in between, ignoring data in the "center",
# ensuring that there is no overlap between
# the two sets caused by the windowing strategy
half = int(n_samples / 2)
offset = int(window / 2) + 2
# using the offset, we ensure that train and val stay INDEPENDENT
x_train = x_sig[:half - offset]
y_train = y_full[:half - offset]
x_val = x_sig[half + offset:]
y_val = y_full[half + offset:]
assert (n_samples - len(x_train) - len(x_val) == 2*offset)

# Track also the starting values from training and validation
# as well as the original chunks (NOT augmented / NOT logreturn)
starting_values_train = starting_values[:half - offset]
starting_values_val = starting_values[half + offset:]
x_win_train = x_win[:half - offset]
x_win_val = x_win[half + offset:]
y_vanilla_train = y_vanilla[:half - offset]
y_vanilla_val = y_vanilla[half + offset:]

print(f"Train data: {x_train.shape}")
print(f"Validation data: {x_val.shape}")
print(f"(original series of lenght {len(time_series)}, window is {window})")
msg()

len_train = len(y_train)
len_val = len(y_val)
plt.axvline(x=window, color="red", linestyle="dashed")
plt.plot(range(window, window + len_train),
		y_vanilla_train,
		label="train values", color="green")
plt.plot(range(window + len_train + 2*offset, 
		window + 2*offset + len_val + len_train),
		y_vanilla_val,
		label="val values", color="red")
plt.plot(range(window, window + n_samples),
		y_vanilla,
		label="full dataset", linestyle="dotted", color="black")
plt.axvline(x=len_train + window, color="grey", linestyle="dashed")
plt.axvline(x=len_train+2*offset+window, color="grey", linestyle="dashed",
		label=f"break of {2*offset}")
plt.grid()
plt.legend()
plt.title(f"Showing training and validation y-data (win = {window})")
plt.show()

# Show some random signatures, just to check that the shapes are bounded
selected_paths = torch.randint(n_samples, (4,))
for nth in selected_paths:
	plot_signature(x_sig[nth], 2, sig_depth)


# DA QUI

##############################################################
####	Here it comes the BAYESIAN APPROACH
#############################################################
D = x_train.shape[1]
n = x_train.shape[0]
X = x_train.T

sigma_noise = 1.
# Prior covariance matrix, selected to be just 1 on the diagonal
bigsigma = torch.eye(D)
i_bigsigma = torch.inverse(bigsigma)
A = torch.matmul(X, X.T) / (sigma_noise ** 2.) + i_bigsigma
i_A = torch.inverse(A)

y = y_train.reshape(n, 1)

# Parameters for the Bayesian regression
tmp1 = torch.matmul(i_A, X)
C = torch.matmul(tmp1, y) / (sigma_noise ** 2)



########
# Predict on TRAIN data
raw_train_predicted = torch.matmul(x_train, C).reshape(-1)

n_pred = x_train.shape[0]
train_uncertainty = torch.zeros(n_pred)
for nth in range(n_pred):
	x_star = x_train[nth].reshape(D, 1)
	prediction_variance = torch.matmul(torch.matmul(x_star.T, i_A), x_star)
	train_uncertainty[nth] = torch.sqrt(prediction_variance)

# Now it is time to predict

pred_train = logback(raw_train_predicted, starting_values_train, final_rescaler)
up_pred_train = logback(raw_train_predicted + 3. * train_uncertainty,
				starting_values_train, final_rescaler)
dw_pred_train = logback(raw_train_predicted - 3. * train_uncertainty,
				starting_values_train, final_rescaler)

train_acc = 0.
for nth in range(n_pred):
	if y_vanilla_train[nth] >= dw_pred_train[nth]:
		if y_vanilla_train[nth] <= up_pred_train[nth]:
			train_acc += 1
train_acc = train_acc * 100. / n_pred
print(f"Train ACC: {train_acc : .2f}")

# Perform a PREDICTION on the VALIDATION data

raw_val_predicted = torch.matmul(x_val, C).reshape(-1)

n_pred = x_val.shape[0]
val_uncertainty = torch.zeros(n_pred)
for nth in range(n_pred):
	x_star = x_val[nth].reshape(D, 1)
	prediction_variance = torch.matmul(torch.matmul(x_star.T, i_A), x_star)
	val_uncertainty[nth] = torch.sqrt(prediction_variance)

# Now it is time to predict

pred_val = logback(raw_val_predicted, starting_values_val, final_rescaler)
up_pred_val = logback(raw_val_predicted + 3. * val_uncertainty,
				starting_values_val, final_rescaler)
dw_pred_val = logback(raw_val_predicted - 3. * val_uncertainty,
				starting_values_val, final_rescaler)


val_acc = 0.
for nth in range(n_pred):
	if y_vanilla_val[nth] >= dw_pred_val[nth]:
		if y_vanilla_val[nth] <= up_pred_val[nth]:
			val_acc += 1
val_acc = val_acc * 100. / n_pred
print(f"Val ACC: {val_acc : .2f}")


# Predicing the NEXT FUTURE value?
# lc here stands for "last_chunk"...
lc_logret = torch.log(last_chunk / last_chunk[0]) * final_rescaler
lc_augment = augment(lc_logret.reshape(-1, 1))
lc_sig = my_signature(lc_augment.unsqueeze(0), depth = sig_depth)

raw_next = torch.matmul(lc_sig, C)
tmp = torch.matmul(lc_sig, i_A)
uncertainty_next = torch.matmul(tmp, lc_sig.T)

next_pt = torch.exp(raw_next / final_rescaler) * last_chunk[0]
up_pt = torch.exp((raw_next+3.*uncertainty_next) /final_rescaler)*last_chunk[0]
dw_pt = torch.exp((raw_next-3.*uncertainty_next) /final_rescaler)*last_chunk[0]


##############################################################################
##	Plot all the results together
##############################################################################
# Predicted train and validation data (must be converted back from logreturn)

plt.axvline(x=window, color="red", linestyle="dashed")
plt.plot(range(window, window + len_train), 
	y_vanilla_train,
	label="train: true", color="green")
plt.plot(range(window, window + len_train), 
	pred_train, label="train: pred", color="red")
plt.plot(range(window, window + len_train), 
	up_pred_train, color="red", linestyle="dotted")
plt.plot(range(window, window + len_train), 
	dw_pred_train, color="red", linestyle="dotted")

plt.axvline(x=window + len_train, color="grey", linestyle="dashed")

plt.plot(range(window + len_train + 2*offset, 
	window + len_val + len_train + 2*offset),
	y_vanilla_val,
	label="val: true", color="green")

plt.plot(range(window + len_train + 2*offset, 
	window + len_val + len_train + 2*offset),
	pred_val, label="val: conditionally pred", color="red")
plt.plot(range(window + len_train + 2*offset, 
	window + len_val + len_train + 2*offset),
	up_pred_val, color="red", linestyle="dotted")
plt.plot(range(window + len_train + 2*offset, 
	window + len_val + len_train + 2*offset),
	dw_pred_val, color="red", linestyle="dotted")

up_pt = up_pt.item()
dw_pt = dw_pt.item()

plt.scatter(len(time_series),next_pt, color = "blue", marker = "o")
plt.scatter(len(time_series),up_pt, color = "blue", marker = "_")
plt.scatter(len(time_series),dw_pt, color = "blue", marker = "_")


plt.axvline(x=window+len_train+2*offset, color="grey", linestyle="dashed")
data_name = sys.argv[1].split("/")[1].split(".")[0]
plt.grid()
plt.legend()
now = time_series[-1]
plt.title(f"[{data_name} win {window}] CURR {now:.1f} NEXT [{up_pt:.1f},{dw_pt:.1f}]")
plt.show()

#####

