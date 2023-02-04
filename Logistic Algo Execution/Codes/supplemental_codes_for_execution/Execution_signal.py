import pandas as pd
import datetime as dt
import numpy as np

class execution:
    def __init__(self, data, signal_return):  # signal_return is the expected return based on signal strength
        self.data = data
        # data.time = data.time.apply(lambda x: dt.datetime.strptime(x, "%d/%m/%Y %H:%M"))
        data.time = pd.to_datetime(data.time, format='%d/%m/%Y %H:%M')  # data prepossessing
        # data.time = pd.to_datetime(data.time, format='%Y-%m-%d %H:%M')  # data prepossessing
        self.signal_expreturn = signal_return
        self.CDF = None  # historical CDF of trading volume
        self.PDF = None #historical PDF of trading volume
        self.execution_percent = []  # record execution percent over time
        self.execution_PDF=[]
        self.market_impact = []  # record market impact over time
        self.signal_alpha = []  # Excess return compared with VWAP based on signal.
        self.vwap_price=[]
        self.total_trade = 200000

    def init(self):
        self.execution_percent = []  # record execution percent over time
        self.execution_PDF = []
        self.market_impact = []  # record market impact over time
        self.signal_alpha = []  # Excess return compared with VWAP based on signal.
        self.vwap_price = []

    def loc_return(self, direction=1.0, signal=0.6):#define expected return
        loc = int(signal * 10) - 5
        if direction == 1.0:
            return self.signal_expreturn[0][loc]
        else:
            return self.signal_expreturn[1][loc]

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
        CDF = df.volume.cumsum()  # CDF of volume
        CDF = CDF / CDF.values[-1]
        #CDF = CDF * 100
        self.CDF = CDF
        self.PDF = CDF.diff()
        self.PDF[0] = self.CDF[0]
        return CDF

    def _market_impact(self,mean_price,high_price,volume_totrade,volume_market:"volume on the market",alpha=1):
        #alpha : price sensitivity
        price_high = mean_price+alpha*(volume_totrade/volume_market)*(high_price-mean_price) # highest trading price
        price_low = mean_price# lowest(passive) trading price
        price_change = (price_high+price_low)/2-mean_price
        return price_change #slippage percentage

    def _volume_totrade(self,market_data,alpha:'price sensitivity',expected_slippage = 0.00002,tolerance = 0.03):#decide how many we should trade
        if market_data['direction'] == 1:#trade aggressively if we are buying
            signal = market_data.logistic_prob# signal strength
            expected_return = self.loc_return(1,signal)# define expected return
            optimal_volume = market_data['average_price']*(expected_return+expected_slippage)/alpha*market_data['volume']/(market_data['high']-market_data['average_price'])
            if len(self.execution_percent)!=0:  #### avoid index out of range 
                max_trade_percent = min(self.CDF[dt.datetime.strftime(market_data.time, "%H %M")]*(1+tolerance),1)-self.execution_percent[-1]
            else:
                max_trade_percent = min(self.CDF[dt.datetime.strftime(market_data.time, "%H %M")] * (1+tolerance), 1) - \
                                    0
            volume= min(self.total_trade*max_trade_percent,optimal_volume)
            volume= max(0,volume)
        elif market_data['direction'] == -1:#trade passively if we are buying
            signal = market_data.logistic_prob # signal strength
            expected_return = -self.loc_return(-1, signal)  # define expected return !negative sign!
            optimal_volume = max(0,market_data['average_price']*(expected_return+expected_slippage)/alpha*market_data['volume']/(market_data['high']-market_data['average_price']))
            if len(self.execution_percent) != 0:  #### avoid index out of range 
                min_trade_percent = min(self.CDF[dt.datetime.strftime(market_data.time, "%H %M")]*(1-tolerance),1)-self.execution_percent[-1]
            else:
                min_trade_percent = min(self.CDF[dt.datetime.strftime(market_data.time, "%H %M")] *(1-tolerance), 1) - 0
            volume = max(self.total_trade*min_trade_percent,optimal_volume)
            volume = max(0, volume)
        else:# no signal, follow VWAP
            optimal_volume = self.total_trade*self.PDF[dt.datetime.strftime(market_data.time, "%H %M")]
            volume = optimal_volume
            volume = max(0, volume)
        return volume

    def _signal_alpha(self):
        signal_alpha = None
        return signal_alpha

    def _VWAP(self, date: "%Y-%m-%d"):########VWAP strats
    #######Sequantially update the self.execution_percent self.market_impact = None self.signal_alpha = None
        self.init()######clear history record
        data =self.data
        data = data.loc[:,['time','open','high','low','close','volume','average_price','logistic_prob','direction']]
        date = dt.datetime.strptime(date,"%Y-%m-%d")
        #data = data.where((data.time >= dt.date(2018, 12, 18)) & (data.time < dt.date(2018, 12, 19)))
        print("tradingday is :",date)
        data = data.where( (data.time >= pd.Timestamp(date.date())) & (data.time < pd.Timestamp(date.date()+dt.timedelta(days=1))) )
        data = data.dropna()
        # print(len(data.time))
        assert len(data.time) == len(self.CDF)######make sure the time period is the same
        for index,market_data in data.iterrows():
            tradevolume = self._volume_totrade(market_data,1)
            price_change = self._market_impact(market_data['average_price'],market_data['high'],tradevolume,market_data['volume'])
            self.market_impact.append(price_change*tradevolume)
            self.execution_PDF.append(tradevolume / self.total_trade)
            self.execution_percent.append(sum(self.execution_PDF))
            ###############compute signal alpha
            avg_tradeprice = np.dot(data.loc[:index,'average_price'],self.execution_PDF)/self.execution_percent[-1]
            vwap_price = np.dot(self.PDF[:dt.datetime.strftime(market_data.time, "%H %M")],data.loc[:index,'average_price'])/self.PDF[:dt.datetime.strftime(market_data.time, "%H %M")].sum()
            self.vwap_price.append(vwap_price)
            # print('avg_tradeprice:',avg_tradeprice)
            # print('vwap_price',vwap_price)
            self.signal_alpha.append(self.total_trade*self.execution_percent[-1]*(-avg_tradeprice+vwap_price))

        return None

    # def result_summary(self):#####present result, graph etc
        # return None


'''
data_test = data_test.loc[:,['time','open','high','low','close','volume','average_price','logistic_prob','direction']]
data_test = data_test.where((data_test.time >= dt.date(2018, 12, 18)) & (data_test.time < dt.date(2018, 12, 19)))
data_test=data_test.dropna()
data_test['volume'].sum()
'''
