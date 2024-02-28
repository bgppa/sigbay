'''
This script is a collection of ways to GENERATE time series, on which a 
prediction will be made by using predict.py
'''
import torch
import matplotlib.pyplot as plt
import sys
from Historic_Crypto import HistoricalData
import datetime
import yfinance as yf
#from libgbm import expected_path, gbm_from_loggbm
#from libgbm import loggbm_from_bm, simple_bm

torch.manual_seed(0)
DATAFOLDER = "./data/"
FORMAT = ".data"


############################################################################
##	TOY DETERMINISTIC EQUATIONS
############################################################################

def exponential_sin (n_times, filename):
	'''
	Generating an exponential sin
	'''
	tmp = torch.linspace(0., 1., n_times)
	# Here the full time series
	time_series = torch.exp(torch.sin(10. * torch.pi * tmp) + 2*tmp)
	# And here the reduced one, on which we will perform the predictions
	torch.save(time_series, DATAFOLDER + filename + FORMAT)
	print(f"dataset of duration {n_times} saved into {filename}") 
	plt.plot(time_series, color="blue")
	plt.grid()
	plt.title("exponential sin")
	plt.show()	
	return time_series
#---

def gen_expsin (name, n_times):
	'''
	Generating an exponential sin
	'''
	tmp = torch.linspace(0., 1., n_times)
	time_series = torch.exp(torch.sin(10. * torch.pi * tmp) + 2*tmp)
	start_date = "0"
	end_date = str(n_times -1)
	return (time_series, "expsin", start_date, end_date)
#---



def simple_polynomial(n_times, filename):
	tmp = torch.linspace(0., 2., n_times)
	time_series = tmp**3 - (2 * (tmp ** 2)) + tmp + 2.
	# And here the reduced one, on which we will perform the predictions
	torch.save(time_series, DATAFOLDER + filename + FORMAT)
	print(f"dataset of duration {n_times} saved into {filename}") 
	plt.plot(time_series, color="blue")
	plt.grid()
	plt.title("simple polynomial")
	plt.show()	
	return time_series
#---

def constant_five (n_times, filename):
	# Here the full time series
	time_series = torch.ones(n_times) * 5.
	# And here the reduced one, on which we will perform the predictions
	torch.save(time_series, DATAFOLDER + filename + FORMAT)
	print(f"dataset of duration {n_times} saved into {filename}") 
	plt.plot(time_series, color="blue")
	plt.grid()
	plt.title("constant value of 5")
	plt.show()	
	return time_series
#--

def cosine_highfreq (n_times, filename):
	'''
	Generating a cosine wave of high frequency
	'''
	tmp = torch.linspace(0., 1., n_times)
	# Here the full time series
	time_series = torch.cos(20. * torch.pi * tmp) + 10
	torch.save(time_series, DATAFOLDER + filename + FORMAT)
	print(f"dataset of duration {n_times} saved into {filename}") 
	plt.plot(time_series, color="blue")
	plt.grid()
	plt.title("cosine with high frequency")
	plt.show()	
	return time_series
#---


#############################################################################
####	From Mathematical Biology
#############################################################################

def gen_gompertz (name, n_times, end_time, N0, NI, b):
	'''
	Generate the Gompertz curve typical for population biology
	'''
	tmp = torch.linspace(0., end_time, n_times)
	factor_2 = (1. - torch.exp(-b * tmp))
	factor_1 = torch.log(torch.tensor(NI / N0))
	time_series = N0 * torch.exp(factor_1 * factor_2)
	start_date = "0"
	end_date = str(end_time)
	return (time_series, "gompertz", start_date, end_date)
#--

def gompertz (n_times, end_time, N0, NI, b):
	'''
	Generating a Gompertz curve
	'''
	filename = "gomp"
	tmp = torch.linspace(0., end_time, n_times)
	factor_2 = (1. - torch.exp(-b * tmp))
	factor_1 = torch.log(torch.tensor(NI / N0))
	time_series = N0 * torch.exp(factor_1 * factor_2)
	# And here the reduced one, on which we will perform the predictions
	torch.save(time_series, DATAFOLDER + filename + FORMAT)
	print(f"dataset of duration {n_times} saved into {filename}") 
	plt.plot(time_series, color="blue")
	plt.grid()
	plt.title("gompertz curve")
	plt.show()	
	return time_series
#---




#############################################################################
##	STOCHASTIC PROCESSES
#############################################################################


def just_uniform(n_times, filename):
	'''
	Generating uniform reals between 1 and 3.
	A good prediction should be 2, right?
	'''
	tmp = torch.linspace(0., 1., n_times)
	# Here the full time series
	time_series = torch.rand((n_times,)) * 2 + 1
	# And here the reduced one, on which we will perform the predictions
	torch.save(time_series, DATAFOLDER + filename + FORMAT)
	print(f"dataset of duration {n_times} saved into {filename}") 
	plt.plot(time_series, color="blue")
	plt.grid()
	plt.title("uniform in [1, 3]")
	plt.show()	
	return time_series
#---


def gbm (n_times, mu, sigma, filename):
	brownian_motion = simple_bm(n_times)
	s0 = 1.0
	logreturns_gbm = loggbm_from_bm(brownian_motion, mu, sigma)
	time_series = gbm_from_loggbm(logreturns_gbm, s0)
	torch.save(time_series, DATAFOLDER + filename + ".data")
	print(f"dataset of duration {n_times} saved into {filename}.data") 
	avg_path = expected_path(n_times, mu, s0)
	torch.save(avg_path, DATAFOLDER + filename + ".avg")
	print(f"Since it is a stochastic process, the expected path")
	print(f"of duration {n_times} is saved into {filename}.avg") 
	plt.plot(time_series, color="blue", label="gbm")
	plt.plot(avg_path, color="orange", linestyle="dashed", label="avg")
	plt.grid()
	plt.title(f"gbm (mu: {mu}, sigma {sigma}) with its expected path")
	plt.show()	
	return time_series
#---


def gaussian_walk(n_times, filename, drift = 0., sigma = 1.):
	'''
	Simple simulation of a Gaussian random walk.
	THIS IS A MARTINGALE
	'''
	results = torch.zeros(n_times)
	for nth in range(n_times):
		results[nth] = results[nth - 1]+torch.normal(drift, sigma,(1,))
	# Translate the results so to have for sure a positive path
	shift = torch.abs(torch.max(results))
	time_series = results + shift
	torch.save(time_series, DATAFOLDER + filename + ".data")
	print(f"dataset of duration {n_times} saved into {filename}.data") 
	plt.plot(time_series, color="blue", label="gaussian walk")
	plt.title(f"Gaussian Random Walk with drift {drift}")
	plt.grid()
	plt.show()
	return time_series
#---

	
def just_gaussians(n_times, filename):
	'''
	Differently from the previous function, here just collection of
	gaussians - this is NOT a Martingale.
	'''
	sigma = 1.
	results = torch.zeros(n_times)
	for nth in range(n_times):
		results[nth] = torch.normal(0., sigma, (1,))
	# Translate the results so to have for sure a positive path
	shift = torch.abs(torch.max(results))
	time_series = results + shift
	torch.save(time_series, DATAFOLDER + filename + ".data")
	print(f"dataset of duration {n_times} saved into {filename}.data") 
	plt.plot(time_series, color="blue", label="gaussian collection")
	plt.grid()
	plt.title("Just a collection of independent gaussians.")
	plt.show()
	return time_series
#---



###########################################################################
####	CRYPTOCURRENCIES
###########################################################################

def crypto_day(token, filename, ndays):
	'''
	Closed value for the ethereum crypto, taken every DAY between
	yesterday and yesterday-DAY days ago.
	Choosing the CLOSE values, so that the intention is to predict
	the next close value.
	'''
	now = datetime.datetime.now()
	delta_start = datetime.timedelta(days = ndays)
	delta_end = datetime.timedelta(days = 1)

	ago = now - delta_start
	start_date = f"{ago.year}-{ago.month}-{ago.day}"
	start_date_fl = start_date + "-00-00"

	one_day = now - delta_end
	end_date = f"{one_day.year}-{one_day.month}-{one_day.day}"
	end_date_fl = end_date + "-23-59"

	ready = HistoricalData(f"{token}-EUR",86400,start_date_fl,end_date_fl)
	new = ready.retrieve_data()

	n_times = new.shape[0]
	time_series = torch.zeros(n_times)
	for nth in range(n_times):
		time_series[nth] = new["close"][nth]
	# And here the reduced one, on which we will perform the predictions
	torch.save(time_series, DATAFOLDER + filename + FORMAT)
	print(f"dataset of duration {n_times} saved into {filename}") 
	print(f"-> HERE: Last value is: {time_series[-1]:.1f}")

	plt.plot(time_series, color="blue")
	plt.title(f"{token}: [{start_date}, {end_date}], {ndays} days.")
	plt.grid()
	plt.show()	
	return time_series
#---


#####################################################################
# STOCKS - YAHOO FINANCE - CLOSE - AMERICAN MARKET AND TIMEZONE
#####################################################################

def yahoo_finance (token, filename, my_period="5y"):
	'''
	For the time period, valid options are for example:
	1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y.
	Can also use datetime to specify a start and an end date.
	'''
	yfdata = yf.Ticker(token)
	stock_values = yfdata.history(period=my_period)
	n_times = len(stock_values)
	time_series = torch.zeros(n_times)
	start_date = ""
	end_date = ""
	for nth in range(n_times):
		time_series[nth] = stock_values["Close"][nth]
	start_t = stock_values.iloc[0].name
	end_t = stock_values.iloc[-1].name

	start_date = f"{start_t.year}-{start_t.month}-{start_t.day}"
	end_date = f"{end_t.year}-{end_t.month}-{end_t.day}"
	torch.save(time_series, DATAFOLDER + filename + FORMAT)
	print(f"dataset of duration {n_times} saved into {filename}") 
	plt.plot(time_series, color="blue")
	plt.grid()
	plt.title(f"{token}[{start_date},{end_date}] {my_period} {n_times}days")
	plt.show()
	return time_series
#---


############################################################################
###	Some utilities to manipulate data
############################################################################

def to_consecutive_relerr (og_series):
	'''
	Take a time series and return the series given by its consecutive
	ratios.
	'''
	n_times = len(og_series)
	result = torch.zeros(n_times - 1)
	for nth in range(1, n_times):
		tmp = torch.abs(og_series[nth] - og_series[nth - 1]) * 100.
		result[nth - 1] = tmp / torch.abs(og_series[nth])
	return result
#---


def to_windowed_mean (og_series, window):
	'''
	Take a time series and return the series given by its consecutive
	ratios.
	'''
	n_times = len(og_series) - window + 1
	result = torch.zeros(n_times)
	for nth in range(n_times):
		curr_window = og_series[nth : nth + window]
		result[nth] = torch.mean(curr_window)
	return result
#---


###########################################################################
#####	Utilities to visualize and store already generated data
###########################################################################

def to_file (time_series, filename):
	n_times = len(time_series)
	torch.save(time_series, DATAFOLDER + filename + FORMAT)
	print(f"dataset of duration {n_times} saved into {filename}") 
	return 1
#---

def time_plot (time_series, name, start_date, end_date):
	n_times = len(time_series)
	plt.plot(time_series, color="blue")
	plt.grid()
	plt.title(f"{name}[{start_date},{end_date}] {n_times} observations")
	plt.show()
	return 1
#---




if __name__ == "__main__":
	
#	ts, name, start_date, end_date = gen_expsin("expsin", 10)
#	to_file(ts, name)
#	time_plot(ts, name, start_date, end_date)
	
#	name = "gompertz"
#	n_times = 2000
#	end_time = 1.6
#	N0 = 0.1
#	NI = 1.
#	b = 2.
#	ts, name, start_date, end_date = gen_gompertz(name, n_times, end_time,
								#N0, NI, b)
#	time_plot(ts, name, start_date, end_date)
#	to_file(ts, name)

###	DETERMINISTIC DATA
#	esin = exponential_sin (1000, "expsin")
#	gompertz(1000, 1.6, 0.1, 1., 2)
#def gompertz (n_times, end_time, N0, NI, b):
#	exponential_sin(25, "expsin-25")
#	simple_polynomial(30, "poly-30")
#	poly = simple_polynomial(500, "poly-500")
#	constant_five(200, "const5")
#	cosine_highfreq(300, "coshf")

###	STOCHASTIC PROCESSES
#	gbm(500, 1.0, 2.7, "gbm500")
#	just_uniform(600, "uniform13")

	
###	CRYPTOCURRENCIES
#	crypto_day("ETH", "eth-days-600", 600)
#	crypto_day("BTC", "btc-days-600", 600)

###	STOCKS from Yahoo Fiance, in dollars, American time zone
#	La mattina in Europa ho il CLOSE americano del giorno prima
#	America = Germania - 9 
#	yahoo_finance ("NTDOY", "nintendo-600d", "600d")
#	yahoo_finance ("MSFT", "microsoft-600d", "600d")
#	yahoo_finance ("NVDA", "nvidia-600d", "600d")
#	yahoo_finance ("PONGF", "atari-600d", "600d")
#	gaussian_walk(500, "gwalk-05-500", drift = 0.5)
#	w2 = gaussian_walk(500, "gwalk-2-500", drift = 2)
	w03 = gaussian_walk(500, "gwalk-03-s9-500", drift = 0.3, sigma = 9.)
#	gaussian_walk(500, "gwalk-01-500", drift = 0.1)
#	w001 = gaussian_walk(500, "gwalk-001-500", drift = 0.01)

#g = just_gaussians(500, "gaussians-500")
