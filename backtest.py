import backtrader as bt
import backtrader.indicators as btind

import talib as ta

from datetime import datetime
import pandas as pd
import numpy as np
import pickle

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV


class MlStrategy(bt.Strategy):

    def __init__(self):
        # Keep a reference to the "close" line in the data[0] dataseries
        self.dataclose = self.datas[0].close

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
        # พอแท่งใหม่ ก็เอาราคาเก่าใส่ข้างล่าง แล้วเติมของใหม่ข้างบน
        #self.log('Close, %.2f' % self.dataclose[0])
        #print('current rsi: {}'.format(self.rsi[0]))
        #print('previous rsi: {}'.format(self.rsi[-1]))

        predict_arr = np.array([np.log(self.dataclose[0]/self.dataclose[-1]),
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

        y_pred = model.predict(predict_arr)

        #print(y_pred)
        if y_pred:
            self.order = self.order_target_size(target=100)
        else:
            self.order = self.order_target_size(target=-100)

    def notify_trade(self, trade):
        if not trade.isclosed:
            return
        self.log('OPERATION PROFIT, GROSS %.2f, NET %.2f' %
                 (trade.pnl, trade.pnlcomm))

    def log(self, txt, dt=None):
        dt = dt or self.datas[0].datetime.date(0)
        print('%s, %s' % (dt.isoformat(), txt))

def clean_dataset(df):
    assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
    df.dropna(inplace=True)
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
    return df[indices_to_keep].astype(np.float64)

if __name__ == '__main__':

    df = pd.read_csv('EURUSD_D1.csv', names=['Date', 'Time', 'Open', 'High', 'Low', 'Close', 'Volume'])
    df['Datetime'] = pd.to_datetime(df.Date + ' ' + df.Time)
    df = df.drop(['Date', 'Time'], axis=1)

    df = df.dropna()
    df = df.set_index('Datetime')

    df = clean_dataset(df)

    start = datetime(2014, 1, 1)
    df = df.loc[start:]

    df.isnull().sum()

    data = bt.feeds.PandasData(dataname=df)

    with open('/Users/ballmdr/knn_data2_D1.pickle', 'rb') as file:
        model = pickle.load(file)

    cerebro = bt.Cerebro()
    cerebro.adddata(data)
    cerebro.addstrategy(MlStrategy)
    cerebro.broker.setcash(1000.0)

    print('Starting Port: %.2f' % cerebro.broker.getvalue())

    results = cerebro.run()

    print('Final Port: %.2f' % cerebro.broker.getvalue())


    cerebro.plot(style='bar')
