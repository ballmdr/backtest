
#%%
import pandas as pd
from datetime import datetime
import numpy as np
import talib as ta


#%%
# Data 3

filename = 'data/EURUSD_H1.csv'
df = pd.read_csv(filename, names=['Date', 'Time', 'Open', 'High', 'Low', 'Close', 'Volume'])
df['Datetime'] = pd.to_datetime(df.Date + ' ' + df.Time)
df = df.drop(['Date', 'Time'], axis=1)
df = df.set_index('Datetime')
df = df.dropna()

df['Target'] = np.where(df.Close.shift(-1) > df.Close, 1, 0)


df['Linear_regression'] = ta.LINEARREG(df.Close, timeperiod=14)
df['Linear_angle'] = ta.LINEARREG_ANGLE(df.Close, timeperiod=14)
df['Linear_slope'] = ta.LINEARREG_SLOPE(df.Close, timeperiod=14)
df['Linear_intercept'] = ta.LINEARREG_INTERCEPT(df.Close, timeperiod=14)

# features
df['body_candle'] = df.Open - df.Close
df['high_low'] = df.High - df.Low
macd, macdsignal, macdhist = ta.MACD(df['Close'].values, fastperiod=12, slowperiod=26, signalperiod=9)
df['macd'] = macd
df['macdsignal'] = macdsignal
df['macdhist'] = macdhist
df['macd-cross'] = np.where(df['macdsignal'] > df['macd'], 1, -1)
df['ma35'] = ta.SMA(df['Close'].values, timeperiod=35)
df['range_ma35'] = df.Close - df.ma35
df['ma35_valid'] = np.where(df.Close >= df.ma35, 1, 0)
df['ma200'] = ta.SMA(df['Close'].values, timeperiod=200)
df['ma200_valid'] = np.where(df.Close >= df.ma200, 1, 0)
df['35_200_cross'] = np.where(df.ma35 >= df.ma200, 1, 0)
df['Returns'] = np.log(df.Close/df.Close.shift(1))
df['ATR'] = ta.ATR(df['High'].values, df['Low'], df['Close'], timeperiod=14)
df['ATR_diff'] = df.ATR.diff()
df['ADX'] = ta.ADX(df.High, df.Low, df.Close, timeperiod=14)
df['ADX_diff'] = df.ADX.diff()
df['CCI'] = ta.CCI(df.High, df.Low, df.Close, timeperiod=14)
df['CCI_diff'] = df.CCI.diff()
df['MOM'] = ta.MOM(df.Close, timeperiod=10)
df['MOM_diff'] = df.MOM.diff()
df['RSI'] = ta.RSI(df.Close, timeperiod=14)
df['RSI_diff'] = df.RSI.diff()
df['Linear_regression_diff'] = df.Linear_regression.diff()
df['Linear_angle_diff'] = df.Linear_angle.diff()
df['Linear_slope_diff'] = df.Linear_slope.diff()
df['Linear_intercept_diff'] = df.Linear_intercept.diff()

df = df.dropna()
drop_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
drop_cols_2 = ['ma200', 'ma35']
drop_cols_3 = ['Linear_regression', 'Linear_angle', 'Linear_slope', 'Linear_intercept']
drop_cols = drop_cols + drop_cols_2 + drop_cols_3

df = df.drop(drop_cols, axis=1)


#%%
X = df.drop('Target', axis=1).values

#%%
from sklearn.preprocessing import StandardScaler, MinMaxScaler
X_scaled = StandardScaler().fit_transform(X)
X_minmax = MinMaxScaler().fit_transform(X)

#%%
df.to_csv('data/data3_M1.csv')






