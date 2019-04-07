import pandas as pd
import numpy as np

import statsmodels.api as sm

eurusd = pd.read_csv('EURUSD_D1.csv', names=['Date', 'Time', 'Open', 'High', 'Low', 'Close', 'Volume'])
eurusd['Datetime'] = pd.to_datetime('')
