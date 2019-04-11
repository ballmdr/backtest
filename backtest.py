#DATA2

import backtrader as bt
import backtrader.indicators as btind

from datetime import datetime, date
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pickle
from backtrader_plotting import Bokeh
from backtrader_plotting.schemes import Tradimo

import sys


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
        self.ma35 = bt.talib.SMA(self.data, timeperiod=35)
        self.ma200 = bt.talib.SMA(self.data, timeperiod=200)
        self.macd = bt.talib.MACD(self.data, fastperiod=12, slowperiod=26, signalperiod=9)
        self.atr = bt.talib.ATR(self.data.high, self.data.low, self.data.close, timeperiod=14)
        self.adx = bt.talib.ADX(self.data.high, self.data.low, self.data.close, timeperiod=14)
        self.cci = bt.talib.CCI(self.data.high, self.data.low, self.data.close, timeperiod=14)
        self.mom = bt.talib.MOM(self.data.close, timeperiod=10)
        self.rsi = bt.talib.RSI(self.data.close, timeperiod=14)

        self.Linear_regression = bt.talib.LINEARREG(self.data.close, timeperiod=14)
        self.Linear_angle = bt.talib.LINEARREG_ANGLE(self.data.close, timeperiod=14)
        self.Linear_slope = bt.talib.LINEARREG_SLOPE(self.data.close, timeperiod=14)
        self.Linear_intercept = bt.talib.LINEARREG_INTERCEPT(self.data.close, timeperiod=14)


    def next(self):
        if self.stats.broker.value[0] < 500.0:
           print('WHITE FLAG ... I LOST TOO MUCH')
           sys.exit()
        self.log('DrawDown: %.2f' % self.stats.drawdown.drawdown[-1])
        self.log('MaxDrawDown: %.2f' % self.stats.drawdown.maxdrawdown[-1])
        #self.log('Close, %.2f' % self.dataclose[0])
        #print('current rsi: {}'.format(self.rsi[0]))
        #print('previous rsi: {}'.format(self.rsi[-1]))

        # Check if an order is pending ... if yes, we cannot send a 2nd one
        if self.order:
            return

        predict_arr = pd.DataFrame([
            self.data.open[0] - self.data.close[0], #body_candle
            self.data.high[0] - self.data.low[0], #high_low
            self.macd.macd[0], #macd
            self.macd.macdsignal[0], #macdsignal
            self.macd.macdhist[0], #macdhist
            np.where(self.macd.macd[0]>=self.macd.macdsignal[0],1,-1), #macd-cross
            self.data.close[0] - self.ma35[0], #range_ma35
            np.where(self.data.close[0]>=self.ma35[0],1,0), #ma35_valid
            np.where(self.data.close[0]>=self.ma200[0],1,0), #ma200_valid
            np.where(self.ma35[0]>=self.ma200[0],1,0), #35_200_cross
            (self.data.close[0]-self.data.close[-1])/self.data.close[-1], #Returns
            self.atr[0], self.atr[0]-self.atr[-1],
            self.adx[0], self.adx[0] - self.adx[-1],
            self.cci[0], self.cci[0] - self.cci[-1],
            self.mom[0], self.mom[0] - self.mom[-1],
            self.rsi[0], self.rsi[0] - self.rsi[-1],
            self.Linear_regression[0] - self.Linear_regression[-1],
            self.Linear_angle[0] - self.Linear_angle[-1],
            self.Linear_slope[0] - self.Linear_slope[-1],
            self.Linear_intercept[0] - self.Linear_intercept[-1]
        ])
        #print(predict_arr)
        predict_arr = MinMaxScaler().fit_transform(predict_arr.values).reshape(1,-1)
        #print(predict_arr)
        y_pred = model.predict(predict_arr)
        #print(y_pred)
        if not self.position:
            if y_pred == 1:
                # BUY, BUY, BUY!!! (with default parameters)
                self.log('BUY CREATE, %.5f' % self.dataclose[0])
                self.order = self.buy(size=self.p.size)
                self.direction = 'long'
            elif y_pred == 0:
                # SELL, SELL, SELL!!! (with all possible default parameters)
                self.log('SELL CREATE, %.5f' % self.dataclose[0])
                # Keep track of the created order to avoid a 2nd order
                self.order = self.sell(size=self.p.size)
                self.direction = 'short'
        else: #close order
            if y_pred == 1 and self.direction == 'short':
                #close short order
                self.log('Close Order, %.5f' % self.dataclose[0])
                self.order = self.close()
                self.direction = None
            elif y_pred == 0 and self.direction == 'long':
                #close long order
                self.log('Close Order, %.5f' % self.dataclose[0])
                self.order = self.close()
                self.direction = None


    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            # Buy/Sell order submitted/accepted to/by broker - Nothing to do
            return

        # Check if an order has been completed
        # Attention: broker could reject order if not enough cash
        if order.status in [order.Completed]:
            if order.isbuy():
                if self.direction != None:
                    self.log(
                        'BUY EXECUTED, Price: %.5f, Cost: %.5f, Comm %.5f' %
                        (order.executed.price,
                        order.executed.value,
                        order.executed.comm))
                    
            else:  # Sell
                if self.direction != None:
                    self.log('SELL EXECUTED, Price: %.5f, Cost: %.5f, Comm %.5f' %
                            (order.executed.price,
                            order.executed.value,
                            order.executed.comm))

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('Order Canceled/Margin/Rejected')
            sys.exit()

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

    #df = clean_dataset(df)

    split = int(len(df)*0.80)
    df = df[split:]

    #print(df.isnull().sum())

    with open('pickle/EURUSD_dec_final_data3_h1.pickle', 'rb') as file:
        model = pickle.load(file)

    data = bt.feeds.PandasData(dataname=df, timeframe=bt.TimeFrame.Minutes, openinterest=None)

    cerebro = bt.Cerebro(stdstats=False)

    cerebro.broker.setcommission()
    cerebro.adddata(data)
    #cerebro.resampledata(data, timeframe=bt.TimeFrame.Minutes, compression=60)
    
    cerebro.addstrategy(MlStrategy)
    start_cash = 1000.0
    cerebro.broker.setcash(start_cash)
    cerebro.broker.setcommission(leverage=10)
    cerebro.addobserver(bt.observers.Broker)
    cerebro.addobserver(bt.observers.DrawDown)

    cerebro.addanalyzer(bt.analyzers.DrawDown)
    cerebro.addanalyzer(bt.analyzers.Returns)
    cerebro.addanalyzer(bt.analyzers.SharpeRatio)
    cerebro.addanalyzer(bt.analyzers.SQN)
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer)
    #cerebro.addanalyzer(bt.analyzers.PyFolio)


    print('Starting Port: %.2f' % cerebro.broker.getvalue())
    
    results = cerebro.run()
    strat = results[0]
    #print('return {}'.format(strat.analyzers.returns.get_analysis()))
    print("Drawdown", strat.analyzers.drawdown.get_analysis())
    print('Sharp Ratio ', strat.analyzers.sharperatio.get_analysis())
    print('SQN ', strat.analyzers.sqn.get_analysis())
    print('Trade Analyzer ', strat.analyzers.tradeanalyzer.get_analysis())

    print('Final Port: %.2f' % cerebro.broker.getvalue())
    pl = start_cash - cerebro.broker.get_value()
    print('pl: %.2f' % pl)

    cerebro.plot()
    bo = Bokeh()
    bo.plot_result(results)
