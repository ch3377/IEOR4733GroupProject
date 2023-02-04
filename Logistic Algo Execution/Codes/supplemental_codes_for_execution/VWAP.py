import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt


class basic_vwap:
    def __init__(self, data, signal_return):  # signal_return is the expected return based on signal strength
        self.data = data
        # data.time = data.time.apply(lambda x: dt.datetime.strptime(x, "%d/%m/%Y %H:%M"))
        data.time = pd.to_datetime(data.time, format='%d/%m/%Y %H:%M')  # data prepossessing
        # data.time = pd.to_datetime(data.time, format='%Y-%m-%d %H:%M')
        self.signal_expreturn = signal_return
        self.CDF = None  # historical CDF of trading volume
        self.execution_percent = {}  # record execution percent over time
        self.market_impact = {}  # record market impact over time
        self.market_impact_effect = {} #####
        self.signal_alpha = None  # Excess return compared with VWAP based on signal.

    # WARNING: the parser date is manual defined
    def _historical_cdf(self):  ######Computing historical volume CDF
        data = self.data
        # data = data.where((data.time >= pd.Timestamp(year=2018, month=12, day=17, hour=0)) & \
        #                   (data.time <= pd.Timestamp(year=2019, month=1, day=1, hour=0)))
        data = data.where((data.time >= dt.datetime(2018, 12, 17)) & \
                          (data.time <= dt.datetime(2018, 12, 31)))
        data = data.dropna()
        data.head()
        data.tail()
        # data.time = data.time.apply(lambda x: str(x.hour)+" "+str(x.minute))
        data.time = data.time.apply(lambda x: dt.datetime.strftime(x, "%H %M"))
        timestamp = np.unique(data.time)
        # pd.to_datetime(timestamp,format = '%H %M')
        df = data.groupby('time').sum()
        CDF = df.volume.cumsum()  # CDV of volume
        CDF = CDF / CDF.values[-1]
        self.CDF = CDF
        return self.CDF

    def _market_impact(self, mean_price, high_price, volume_totrade, volume_market:"volume on the market", alpha=1):
        # alpha : price sensitivity
        price_high = mean_price+alpha*(volume_totrade/volume_market)*(high_price-mean_price)  # highest trading price
        price_low = mean_price  # lowest(passive) trading price
        price_change = (price_high+price_low)/2 - mean_price
        return price_change  # slippage percentage

    def _volume_totrade(self, time, alpha:'price sensitivity'):  # decide how many we should trade
        volume = self.CDF[time]  # for VWAP
        return volume

    def _signal_alpha(self):
        signal_alpha = None
        return signal_alpha

    def _VWAP(self):########VWAP strats
        #  Sequentially update the self.execution_percent self.market_impact = None self.signal_alpha = None
        s_rate = 1
        day_tradesrs = {}
        day_tradesum = 0
        day_mkt_impactsrs = {}
        day_mkt_impact = 0
        market_impact_effect = {}  #######
        daily_goal = 200000  # DAILY TRADE GOAL

        for index, row in self.data.iterrows():
            if index > 0 and row['time'].date() != self.data.ix[index-1, 'time'].date():
                day_tradesum = 0  # this is the trade sum already happened within one day
                day_mkt_impact = 0  # this is the market impact sum already happened within one day
                self.execution_percent[self.data.ix[index-1, 'time'].date()] = pd.Series(day_tradesrs)
                self.market_impact[self.data.ix[index-1, 'time'].date()] = pd.Series(day_mkt_impactsrs)
                self.market_impact_effect[self.data.ix[index-1, 'time'].date()] = pd.Series(market_impact_effect) #######

                day_tradesrs = {}
                day_mkt_impactsrs = {}
                market_impact_effect = {}  ##########

            time_index = dt.datetime.strftime(pd.to_datetime(row['time'], format='%Y%m%d %H:%M'), "%H %M")

            goal_pct = self._volume_totrade(time_index, None) - day_tradesum  # current goal trade percentage
            trade_pct = s_rate * goal_pct  # current achieved trade percentage
            day_tradesum += trade_pct
            day_tradesrs[time_index] = day_tradesum

            mean_price = np.mean(row['open':'close'])
            # the sign of market_impact
            ##########
            market_impact_effect[time_index] = self._market_impact(mean_price, row['high'], int(trade_pct*daily_goal), row['volume'])
            day_mkt_impact += trade_pct * daily_goal * market_impact_effect[time_index]
            # if data.ix[index, 'time'].date() == dt.date(2018, 10, 1):
            # print(trade_pct * daily_goal)
            day_mkt_impactsrs[time_index] = day_mkt_impact

            if index == self.data.shape[0]-1:
                self.execution_percent[self.data.ix[index-1, 'time'].date()] = pd.Series(day_tradesrs)
                self.market_impact[self.data.ix[index-1, 'time'].date()] = pd.Series(day_mkt_impactsrs)
                self.market_impact_effect[self.data.ix[index-1, 'time'].date()] = pd.Series(market_impact_effect) #######

        return None

    def result_summary(self, T):  #####present result, graph etc
        # T = dt.date(2018, 12, 28)
        # daily execution percentage
        exe_pct = self.execution_percent[T]
        plt.plot(exe_pct)
        plt.xticks(np.arange(0, len(exe_pct), 30), exe_pct.index[0:len(exe_pct):30], fontsize=8)
        plt.xlabel('time')
        plt.ylabel('percentage')
        plt.title('%s Execution Percentage' % T.strftime('%Y-%m-%d'))
        plt.show()
        plt.close()

        # daily market impact loss
        mkt_ipct = self.market_impact[T]
        plt.plot(mkt_ipct)
        plt.xticks(np.arange(0, len(mkt_ipct), 30), mkt_ipct.index[0:len(mkt_ipct):30], fontsize=8)
        plt.xlabel('time')
        plt.ylabel('total loss')
        plt.title('%s Total Market Impact Loss' % T.strftime('%Y-%m-%d'))
        plt.show()
        plt.close()
        # print(mkt_ipct)
        # print(np.diff(mkt_ipct))

        # daily minute price slippage
        mkt_ipct_fct = self.market_impact_effect[T]
        plt.plot(mkt_ipct_fct)
        plt.xticks(np.arange(0, len(mkt_ipct_fct), 30), mkt_ipct_fct.index[0:len(mkt_ipct_fct):30], fontsize=8)
        plt.xlabel('time')
        plt.ylabel('price change')
        plt.title('%s Minute Price Slippage' % T.strftime('%Y-%m-%d'))
        plt.show()
        plt.close()
        print(mkt_ipct_fct)

        return None
##############Test case


# dir_data = "aapl_signal.csv"
# data = pd.read_csv(dir_data)

# VWAP_Class = basic_vwap(data, None)
# print(VWAP_Class._historical_cdf())
# print(VWAP_Class.data.ix[1])

# VWAP_Class._VWAP()
# VWAP_Class.result_summary(dt.date(2018, 12, 17))
# VWAP_Class.result_summary(dt.date(2018, 12, 28))
# VWAP_Class.result_summary(dt.date(2018, 12, 31))
