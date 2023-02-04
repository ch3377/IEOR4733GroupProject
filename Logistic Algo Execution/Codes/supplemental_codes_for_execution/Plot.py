from VWAP import basic_vwap
from Execution_signal import execution
import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt


############## Test case for basic_vwap and execution

# input data
dir_data = "aapl_signal.csv"
data = pd.read_csv(dir_data)

# run vwap and all date results are in VWAP
VWAP = basic_vwap(data, None)
VWAP._historical_cdf()
VWAP._VWAP()

# run strategy and all things ready to play in execution_strategy
r = ([0.00018540385841424813, 0.00019730658105338646, 0.0003623564039380093, 0.0006224806392869604, 0.0006264755945804102], [2.8591412486349406e-05, 0.00017477819656313643, 0.0004727381734393411, 0.0006218296307207129, 0.0007317083530385608])
execution_strategy = execution(data, r)
execution_strategy._historical_cdf()


###########Single-day analysis
# choose the analyzing date
T = dt.date(2018, 12, 28)  # input for vwap class
t = '2018-12-28'  # input for signal class
execution_strategy._VWAP(t)


#####cumulative execution percentage
vwap_pct = VWAP.execution_percent[T]
fig = plt.figure(figsize=(10, 5))
ax1 = fig.add_subplot(111)
ax1.grid(True, linestyle="--", alpha=0.5)
ax1.plot(vwap_pct)
ax1.plot(execution_strategy.execution_percent)
ax1.set_ylabel('percentage')
##minute execution percentage
vwap_vol = pd.Series.diff(vwap_pct)
vwap_vol[0] = vwap_pct[0]
ax2 = ax1.twinx()
ax2.plot(vwap_vol)
ax2.plot(execution_strategy.execution_PDF, alpha=0.8)
ax2.set_ylim([0, 0.1])
ax2.set_ylabel('minute percentage')

# WARNING: not generally applicable time_ticker
time_ticker = np.append(np.array(vwap_pct.index[:]), "16 00")
plt.xticks(np.arange(0, len(time_ticker), 30), time_ticker[0:len(time_ticker):30], fontsize=8)
plt.title('%s Execution Percentage' % T.strftime('%Y-%m-%d'))
plt.legend(labels=['VWAP', 'ML-VWAP'], loc='upper left')
plt.show()
plt.close()

# average stock price series
avg_p = data[['time', 'average_price']]
# print(avg_p)
avg_p.time = avg_p.time.apply(lambda x: x.date())
avg_p_gp = avg_p.groupby('time')
avg_pt = avg_p_gp.get_group(dt.date(2018, 12, 28))


#####stratege overview
fig = plt.figure(figsize=(10, 5))
ax1 = fig.add_subplot(111)
ax1.grid(True, linestyle="--", alpha=0.5)
ax1.plot(vwap_vol)
ax1.plot(execution_strategy.execution_PDF, alpha=0.8)
ax1.set_ylim([0, 0.1])
ax1.set_ylabel('minute percentage')
ax1.legend(labels=['VWAP', 'ML-VWAP'], loc='upper left')

ax2 = ax1.twinx()
ax2.plot(VWAP.execution_percent[T].index, avg_pt['average_price'], alpha=0.6, color='g')
ax2.legend(labels=['Stock Price'], loc='upper center')
ax2.set_ylabel('price')

plt.xticks(np.arange(0, len(time_ticker), 30), time_ticker[0:len(time_ticker):30], fontsize=8)
plt.title('%s Minute Execution Percentage' % T.strftime('%Y-%m-%d'))
plt.show()
plt.close()


#####market_impact loss
fig = plt.figure(figsize=(9, 5))
ax1 = fig.add_subplot(111)
ax1.grid(True, linestyle="--", alpha=0.5)
ax1.plot(VWAP.market_impact[T])
ax1.plot(np.cumsum(execution_strategy.market_impact))
ax1.set_ylabel('total loss')
ax1.legend(labels=['VWAP', 'ML-VWAP'], loc='best')

ax2 = ax1.twinx()
ax2.plot(VWAP.market_impact[T].index, avg_pt['average_price'], alpha=0.2, color='g')
ax2.legend(labels=['Stock Price'], loc='upper center')
ax2.set_ylabel('price')

plt.xticks(np.arange(0, len(time_ticker), 30), time_ticker[0:len(time_ticker):30], fontsize=8)
plt.xlabel('time')
plt.title('%s Total Market Impact Loss' % T.strftime('%Y-%m-%d'))
plt.show()
plt.close()


#####strategy alpha compared with vwap
fig = plt.figure(figsize=(9, 5))
ax1 = fig.add_subplot(111)
ax1.grid(True, linestyle="--", alpha=0.5)
daily_alpha = execution_strategy.signal_alpha
ax1.plot(daily_alpha)
# print(type(np.diff(daily_alpha)), type(daily_alpha[0]))
ax1.plot([daily_alpha[0]] + np.diff(daily_alpha), alpha=0.8)
ax1.set_ylabel('alpha')
ax1.legend(labels=['Total Alpha', 'Minute Alpha'], loc='best')

ax2 = ax1.twinx()
ax2.plot(VWAP.market_impact[T].index, avg_pt['average_price'], alpha=0.2, color='g')
ax2.legend(labels=['Stock Price'], loc='upper center')
ax2.set_ylabel('price')

plt.xticks(np.arange(0, len(time_ticker), 30), time_ticker[0:len(time_ticker):30], fontsize=8)
plt.xlabel('time')
plt.title('%s Daily Alpha' % T.strftime('%Y-%m-%d'))
plt.show()
plt.close()


# #####plot histogram
# fig = plt.figure(figsize=(4,3))
# ax1 = fig.add_subplot(111)
# ax1.hist(np.abs(data['average_price'].diff()).dropna())
# ax1.set_title('histogram of price change')
# fig.show()
# plt.close()


#####target cdf and real volume cdf
# compute real cdf
data = data.loc[:,['time','open','high','low','close','volume','average_price','logistic_prob','direction']]
date = dt.datetime.strptime('2018-12-28',"%Y-%m-%d")
#data = data.where((data.time >= dt.date(2018, 12, 18)) & (data.time < dt.date(2018, 12, 19)))
data = data.where( (data.time >= pd.Timestamp(date.date())) & (data.time < pd.Timestamp(date.date()+dt.timedelta(days=1))) )
data = data.dropna()
real = np.cumsum(data['volume'])/np.sum(data['volume'])

fig = plt.figure(figsize=(8, 5))
plt.plot(execution_strategy.CDF.tolist())
plt.plot(real.tolist())
plt.legend(["Target Volume", "Real Volume"])
plt.xticks(np.arange(0, len(time_ticker), 30), time_ticker[0:len(time_ticker):30], fontsize=8)
plt.xlabel('time')
plt.ylabel('volume percentage')
plt.title('%s Market Volume Distribution' % T.strftime('%Y-%m-%d'))
plt.show()
plt.close()


###########Period analysis
data_test = execution_strategy.data
tradingday = list(np.unique(data_test.time.apply(lambda x: dt.datetime.strftime(x, '%Y-%m-%d'))))
revenue = []
market_impact = []
days = []
for day in tradingday:
    try:
        execution_strategy._VWAP(day)
        revenue.append(execution_strategy.signal_alpha[-1])
        market_impact.append(sum(execution_strategy.market_impact))
        days.append(day)
    except:
        pass

#####strategy period alphas
fig = plt.figure(figsize=(8, 5))
ax1 = fig.add_subplot(111)
ax1.yaxis.grid(True, linestyle="--", alpha=0.5)
ax1.bar(days, revenue)
plt.xticks(fontsize=8)
plt.xlabel('date')
plt.ylabel('alpha')
# plt.ylim([0, 60000])
plt.title('2018/12/17-12/31 Strategy Daily Alphas')
plt.show()
plt.close()
