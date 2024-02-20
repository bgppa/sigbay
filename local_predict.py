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

TESTDATA = True # if True, discard part of the final series to use as test
EPOCH_UNITS = 3 # each unit here corresponst to 10_000 epochs
FUTURE = 5	# number future points to predict wrt val data size
GIVENAVG = 0	# True if the average path is provided (gbm/stoch proc case)
DEBUG = 0
RESCALER = 3.

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
time_series = torch.load(filename).reshape(-1,)
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
###	Train the linear model on the signature data
###############################################################
lr = 1e-4
model = linear_training(sig_len,EPOCH_UNITS,x_train,y_train,x_val,y_val,lr)
diagonal_test(x_train, y_train, x_val, y_val, model)


##################################################################
###	Predict the future
##################################################################
print(f"Predicting {len_future} FUTURE values...")
future_values = predict_from(last_chunk, model, sig_depth, len_future,
		final_rescaler)

bak_future_values = future_values.clone().detach()


##############################################################################
##	Plot all the results together
##############################################################################
# Predicted train and validation data (must be converted back from logreturn)
pred_train = logback(model(x_train).detach(), starting_values_train,
		final_rescaler)
pred_val = logback(model(x_val).detach(), starting_values_val, final_rescaler)

plt.axvline(x=window, color="red", linestyle="dashed")
plt.plot(range(window, window + len_train), 
	y_vanilla_train,
	label="train: true", color="black")
plt.plot(range(window, window + len_train), 
	pred_train, label="train: pred", color="grey",
	linestyle="dotted")
plt.axvline(x=window + len_train, color="grey", linestyle="dashed")

plt.plot(range(window + len_train + 2*offset, 
	window + len_val + len_train + 2*offset),
	y_vanilla_val,
	label="val: true", color="green")

plt.plot(range(window + len_train + 2*offset, 
	window + len_val + len_train + 2*offset),
	pred_val,
	label="val: conditionally pred", color="red", linestyle="dashdot")
plt.axvline(x=window+len_train+2*offset, color="grey", linestyle="dashed")

# Plot the predicted future values
duration = len(y_full)
assert (duration == len_val + len_train + 2*offset)
plt.axvline(x=duration + window, color="grey", linestyle="dashed")
plt.plot(range(duration + window, window+len_future + duration), future_values,
		label=f"test: blind predicted ({len_future})", color="orange",
		linestyle="dashdot")
plt.axvline(x=duration+window+ len_future, color="grey", linestyle="dashed")
# If test data are available, plot them, too
if TESTDATA:
	plt.plot(range(duration+window, window+len_future+duration), y_test,
		label=f"test: true", color="blue",
		linestyle="dashdot")
if GIVENAVG:
	plt.plot(avg_path, label="avg path",color="purple",linestyle="dashdot")

# Extract the name of the dataset, just for the plot title
data_name = sys.argv[1].split("/")[1].split(".")[0]
plt.grid()
plt.legend()
plt.title(f"sigwzrd2[{data_name} win {window} ep{EPOCH_UNITS}0K ]")
plt.show()

##############################################################
# REVISED Segmented TEST
##############################################################
#end_point = n_samples
end_point = n_times
start_point = end_point - len_future
plt.axvline(x = end_point, color="grey", linestyle="dashed")
while (start_point > window):
	plt.axvline(x = start_point, color="grey", linestyle="dashed")
	plt.plot(range(start_point, end_point),
		# y_vanilla at index "i" contains data from the original
		# time series at time "i + windows"
		# since here we want to visualize the time series data between
		# start and end point, we need the idex translation as shown
		y_vanilla[start_point - window : end_point - window],	
		color = "green")
	if GIVENAVG:
		plt.plot(range(start_point, end_point),
			avg_path[start_point : end_point],
			color = "purple", linestyle="dashdot")

	# Predict the future starting from the corresponding chunk
	# x_win at index "i" contains the time series chunk from time
	# "i" to time "i + window". Therefore at index "start_point - window"
	# it contains data from start_point-win to start_point, i.e. 
	# the chunk useful to predict the value at time "start_point" and then
	# all the following one accoring to the len_future
	curr_chunk = x_win[start_point - window]
	future_values = predict_from(curr_chunk, model, sig_depth, len_future,
				final_rescaler)

	plt.plot(range(start_point, end_point),	future_values, color="orange")
	end_point = start_point
	start_point = end_point - len_future

plt.plot(range(start_point + len_future, n_times),
	y_vanilla[start_point + len_future - window:], color = "blue")
# here start_point becames lower than the observing window


if TESTDATA:
	plt.plot(range(duration + window, window+len_future + duration), 
		bak_future_values,
		label=f"test: blind predicted ({len_future})", color="red")
	plt.plot(range(duration + window, window+len_future + duration),y_test,
			color="green")


plt.title("Hard Segment Test")
shift = int(end_point) % int(len_future)
x_name = f"{end_point} points, predictions {len_future}, shift {shift}"
plt.xlabel(x_name)
plt.grid()
#plt.legend()
plt.show()


#input("Ready for 1-step reliability?")
print(f"Now the 1-step reliability!")


#################################################################
# Reliability TEST
#################################################################
#recent_time = int(x_win_val.shape[0] / 3)
#x_win_recent = x_win_val[-recent_time:]
#y_recent = y_vanilla_val[-recent_time:]
#print(f"*** RELIABILITY TEST ***")

m_train, train_balance = reliability_analysis(x_win_train, y_vanilla_train, 
				model, sig_depth, 1, final_rescaler)
training_reliability = m_train[0][0] + m_train[1][1]
print(f"TRAINING [{len(y_train)}]: {training_reliability : .1f}%")
print(m_train)
print(f"Train true go-up frequency: {train_balance : .2f}%")

m_val, val_balance = reliability_analysis(x_win_val, y_vanilla_val,
		model, sig_depth, 1, final_rescaler)
validation_reliability = m_val[0][0] + m_val[1][1]
print(f"VALIDATION [{len(y_val)}]: {validation_reliability : .1f}%")
print(m_val)
print(f"Validation true go-up frequency: {val_balance : .2f}%")


#m_recent = reliability_analysis(x_win_recent, y_recent,
#		model, sig_depth, len_future)
#recent_reliability = m_recent[0][0] + m_recent[1][1]
#print(f"RECENT [{len(y_recent)}]: {recent_reliability : .1f}%")
#print(m_recent)

#############################################################################
####	Rudimental part on DRIFT DETECTION
#############################################################################

tmp_train = pred_train[1:].reshape(-1)
tmp_truetrain = y_vanilla_train[:-1].reshape(-1)
train_drifts = tmp_train - tmp_truetrain
mean_drift_train = torch.mean(train_drifts)
plt.hist(train_drifts)

tmp_val = pred_val[1:].reshape(-1)
tmp_trueval = y_vanilla_val[:-1].reshape(-1)
validation_drifts = tmp_val - tmp_trueval
mean_drift_val = torch.mean(validation_drifts)
plt.hist(validation_drifts)
plt.title(f"Drift train {mean_drift_train:.3f} val {mean_drift_val:.3f}")


plt.show()


# Predicted drift
b = future_values[:-1]
a = future_values[1:]
print(f"{a - b}")
print(f"stimated drift on FUTURE: {torch.mean(a - b) : .3f}")


