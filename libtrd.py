'''
This is a very simple library when I implement my custom trading
strategies.
'''
import torch


def simple_trading (curr_stocks, curr_budget, true_values,
			up_pred, dw_pred, buy_perc, sell_perc):
	'''
	A very simple strategy: at each time, if the true value stays in
	the predicted range, we do not do nothing.
	If it exceeds the upper confidence interval, we buy.
	If it goes below the bottom confidence interval, we sell.
	'''
	times = len(true_values)
	initial_investment = (curr_stocks*true_values[0] + curr_budget).item()
	for i in range(1, times):
		if true_values[i] > up_pred[i]:
			# buy buy_perc% of stocks
			print(f"time {i}: trying buy {to_buy} stocks")
			to_buy = curr_stocks / 100 * buy_perc
			to_spend = (to_buy * true_values[i]).item()
			if (to_spend <= curr_budget):	
				curr_budget -= to_spend
				curr_stocks += to_buy
				print("OK")
			else:
				print(f"Not enough budget")
		elif true_values[i] < dw_pred[i]:
			# sell sell_perc% of stocks
			to_sell = curr_stocks / 100 * sell_perc
			curr_budget += (to_sell * true_values[i]).item()
			curr_stocks -= to_sell
			print(f"time {i}: sell {to_sell} stocks")
		else:
			# stay quiet
			print(f"time {i}: idle")
		print(f"{i} stocks {curr_stocks:.2e} budget {curr_budget:.3f}")
		total_value = (curr_stocks*true_values[i] + curr_budget).item()
		print(f"total value {total_value:.3f}")

	final_value = (curr_stocks*true_values[-1] + curr_budget).item()
	print(f"Started with {initial_investment:.3f} EUR")
	print(f"Ended with {final_value:.3f} EUR")
	if final_value > initial_investment:
		print("APPROVED")
		return 1
	else:
		print("FAIL")
		return 0
#---


def idle_trading (curr_stocks, curr_budget, true_values):
	'''
	Just an idle strategy: buy and keep idle for the whole range
	of time.
	'''
	initial_investment = (curr_stocks*true_values[0] + curr_budget).item()
	final_value = (curr_stocks*true_values[-1] + curr_budget).item()
	print(f"Started with {initial_investment:.3f} EUR")
	print(f"Ended with {final_value:.3f} EUR")
	if final_value > initial_investment:
		print("APPROVED")
		return 1
	else:
		print("FAIL")
		return 0
#---


def averaged_trading (curr_stocks, curr_budget, true_values,
			n_days, buy_perc, sell_perc):
	'''
	A very simple strategy: at each time,
	I look at the average of the last n days and compare to today.
	If today is higher, I buy.
	If lower, I sell.
	'''
	assert (n_days >= 1)
	times = len(true_values)
	initial_investment = (curr_stocks*true_values[0] + curr_budget).item()
	# I start trading at day n_days;
	# The first day is day 0
	# (therefore I wait n_days, and then start trading)
	for i in range(n_days, times):
		averaged_mean = torch.mean(true_values[i - n_days:i])
		today = true_values[i]
		if today >= averaged_mean:	
			# buy buy_perc% of stocks
			to_buy = curr_stocks / 100 * buy_perc
			curr_budget -= (to_buy * true_values[i]).item()
			curr_stocks += to_buy
			print(f"time {i}: buy {to_buy} stocks")
		else:
			# sell sell_perc% of stocks
			to_sell = curr_stocks / 100 * sell_perc
			curr_budget += (to_sell * true_values[i]).item()
			curr_stocks -= to_sell
			print(f"time {i}: sell {to_sell} stocks")
		print(f"{i} stocks {curr_stocks:.2e} budget {curr_budget:.3f}")
		total_value = (curr_stocks*true_values[i] + curr_budget).item()
		print(f"total value {total_value:.3f}")

	final_value = (curr_stocks*true_values[-1] + curr_budget).item()
	print(f"Started with {initial_investment:.3f} EUR")
	print(f"Ended with {final_value:.3f} EUR")
	if final_value > initial_investment:
		print("APPROVED")
		return 1
	else:
		print("FAIL")
		return 0
#---

