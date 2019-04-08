#DATA2

import backtrader as bt
import backtrader.indicators as btind

from datetime import datetime, date
import pandas as pd
import numpy as np

from keras.models import load_model
from sklearn.preprocessing import StandardScaler
import pyfolio as pf

class MlStrategy(bt.Strategy):

    params = (('size', 1000),)

    def __init__(self):
        # Keep a reference to the "close" line in the data[0] dataseries
        self.dataclose = self.datas[0].close
        # To keep track of pending orders and buy price/commission
        self.order = None
        self.buyprice = None
        self.buycomm = None
        self.direction = None

        # features
        self.adx = bt.talib.ADX(self.data.high, self.data.low, self.data.close, timeperiod=14)
        self.cci = bt.talib.CCI(self.data.high, self.data.low, self.data.close, timeperiod=14)
        self.mom = bt.talib.MOM(self.data.close, timeperiod=10)
        self.rsi = bt.talib.RSI(self.data.close, timeperiod=14)

        self.Linear_regression = bt.talib.LINEARREG(self.data.close, timeperiod=14)
        self.Linear_angle = bt.talib.LINEARREG_ANGLE(self.data.close, timeperiod=14)
        self.Linear_slope = bt.talib.LINEARREG_SLOPE(self.data.close, timeperiod=14)
        self.Linear_intercept = bt.talib.LINEARREG_INTERCEPT(self.data.close, timeperiod=14)


    def next(self):
        #self.log('Close, %.2f' % self.dataclose[0])
        #print('current rsi: {}'.format(self.rsi[0]))
        #print('previous rsi: {}'.format(self.rsi[-1]))

        # Check if an order is pending ... if yes, we cannot send a 2nd one
        if self.order:
            return

        predict_arr = np.array([(self.dataclose[0]-self.dataclose[-1])/self.dataclose[-1],
                                self.adx[0] - self.adx[-1],
                                self.cci[0] - self.cci[-1],
                                self.mom[0] - self.mom[-1],
                                self.rsi[0] - self.rsi[-1],
                                self.Linear_regression[0] - self.Linear_regression[-1],
                                self.Linear_angle[0] - self.Linear_angle[-1],
                                self.Linear_slope[0] - self.Linear_slope[-1],
                                self.Linear_intercept[0] - self.Linear_intercept[-1]
                                ]).reshape(1, -1)

        #print(predict_arr.reshape(1, -1))
        predict_arr = scaler.transform(predict_arr)
        y_pred = model.predict(predict_arr)
        print(y_pred)
        if not self.position:
            if y_pred > 0.5:
                # BUY, BUY, BUY!!! (with default parameters)
                self.log('BUY CREATE, %.5f' % self.dataclose[0])
                self.order = self.buy(size=self.p.size)
                self.direction = 'long'
            elif y_pred < 0.5:
                # SELL, SELL, SELL!!! (with all possible default parameters)
                self.log('SELL CREATE, %.5f' % self.dataclose[0])
                # Keep track of the created order to avoid a 2nd order
                self.order = self.sell(size=self.p.size)
                self.direction = 'short'
        else:
            if y_pred > 0.5 and self.direction == 'short':
                # BUY, BUY, BUY!!! (with default parameters)
                self.log('BUY CREATE, %.5f' % self.dataclose[0])
                self.order = self.buy(size=self.p.size)
                self.direction = None
            elif y_pred < 0.5 and self.direction == 'long':
                # SELL, SELL, SELL!!! (with all possible default parameters)
                self.log('SELL CREATE, %.5f' % self.dataclose[0])
                # Keep track of the created order to avoid a 2nd order
                self.order = self.sell(size=self.p.size)
                self.direction = None


    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            # Buy/Sell order submitted/accepted to/by broker - Nothing to do
            return

        # Check if an order has been completed
        # Attention: broker could reject order if not enough cash
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(
                    'BUY EXECUTED, Price: %.5f, Cost: %.5f, Comm %.5f' %
                    (order.executed.price,
                     order.executed.value,
                     order.executed.comm))

                self.buyprice = order.executed.price
                self.buycomm = order.executed.comm
            else:  # Sell
                self.log('SELL EXECUTED, Price: %.5f, Cost: %.5f, Comm %.5f' %
                         (order.executed.price,
                          order.executed.value,
                          order.executed.comm))

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('Order Canceled/Margin/Rejected')

        self.order = None


    def notify_trade(self, trade):
        if not trade.isclosed:
            return

        self.log('OPERATION PROFIT, GROSS {}'.format(trade.pnl))

    def log(self, txt, dt=None):
        dt = dt or self.datas[0].datetime.date(0)
        print('%s, %s' % (dt.isoformat(), txt))

def clean_dataset(df):
    assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
    df.dropna(inplace=True)
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
    return df[indices_to_keep].astype(np.float64)

if __name__ == '__main__':

    df = pd.read_csv('data/EURUSD_M1.csv', names=['Date', 'Time', 'Open', 'High', 'Low', 'Close', 'Volume'])
    df['Datetime'] = pd.to_datetime(df.Date + ' ' + df.Time)
    df = df.set_index('Datetime')
    df = df.drop(['Date', 'Time'], axis=1)

    df = clean_dataset(df)

    start = date(2019,1,1)
    end = date(2018,1,31)
    df = df.loc[start:]

    #print(df.isnull().sum())

    data = bt.feeds.PandasData(dataname=df, timeframe=bt.TimeFrame.Minutes, openinterest=None)
    model = load_model('h5/keras_data2_M1_53.h5')

    df_scaler = pd.read_csv('data/data2_M1.csv', parse_dates=['Datetime'], index_col='Datetime')
    X = df_scaler.drop('Target', axis=1).values
    scaler = StandardScaler()
    scaler.fit(X)

    cerebro = bt.Cerebro()

    cerebro.broker.setcommission()
    #cerebro.adddata(data)
    cerebro.resampledata(data, timeframe=bt.TimeFrame.Minutes, compression=1440)
    cerebro.addstrategy(MlStrategy)
    cerebro.broker.setcash(1000.0)
    cerebro.addanalyzer(bt.analyzers.PyFolio, _name='pyfolio')
    print('Starting Port: %.2f' % cerebro.broker.getvalue())
    
    results = cerebro.run()

    print('Final Port: %.2f' % cerebro.broker.getvalue())

