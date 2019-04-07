import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

import statsmodels.api as sm

def get_data(filename):
    df = pd.read_csv(filename, names=['Date', 'Time', 'Open', 'High', 'Low', 'Close', 'Volume'])
    df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
    df = df.set_index('Datetime')
    df = df.drop(['Date', 'Time'], axis=1)
    return df

eurusd = get_data('EURUSD_D1.csv')
gbpusd = get_data('GBPUSD_D1.csv')

aapl = pd.read_csv('AAPL.csv')

returns = pd.DataFrame()
returns['EURUSD'] = eurusd.Close.pct_change()
returns['GBPUSD'] = gbpusd.Close.pct_change()
returns['AAPL'] = aapl.Close.pct_change()

x = pd.DataFrame(returns.EURUSD, columns=['X'])
x = sm.add_constant(x)
result = sm.OLS(returns.GBPUSD, x).fit()
print(type(returns.GBPUSD))

print(result.summary())


correlation = returns.EURUSD.corr(returns.GBPUSD)
print('Correlation: {}'.format(correlation))

plt.scatter(returns['EURUSD'], returns['GBPUSD'])
plt.show()
