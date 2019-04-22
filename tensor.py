#%%
import tensorflow as tf 
from tensorflow import keras
import pandas as pd 
import numpy as np
from sklearn.preprocessing import MinMaxScaler, scale, PolynomialFeatures, StandardScaler
from sklearn.decomposition import PCA
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

def plot_history(histories, key='acc'):
  plt.figure(figsize=(16,10))

  for name, history in histories:
    val = plt.plot(history.epoch, history.history['val_'+key],
                   '--', label=name.title()+' Val')
    plt.plot(history.epoch, history.history[key], color=val[0].get_color(),
             label=name.title()+' Train')

  plt.xlabel('Epochs')
  plt.ylabel(key.replace('_',' ').title())
  plt.legend()

  plt.xlim([0,max(history.epoch)])

def create_multivariate_rnn_data(x, y, window_size):
    y = y[window_size:]
    n = x.shape[0]
    x = np.stack([x[i: j] for i, j in enumerate(range(window_size, n))], axis=0)
    return x, y

df = pd.read_csv('data/data3_M1.csv', parse_dates=['Datetime'], index_col='Datetime')
x = df.drop('Target', axis=1).values

x = MinMaxScaler().fit_transform(x)
#x = PolynomialFeatures().fit_transform(x)
#with open('pca21_data4_h1.pickle', 'rb') as file:
#   pca = pickle.load(file)
pca = PCA(n_components=.8)
x = pca.fit_transform(x)
y = df.Target.values

split = int(len(x)*0.75)
split2 = int(len(x)*0.80)
x_train = x[:split]
x_test = x[split2:]
y_train = y[:split]
y_test = y[split2:]

n_cols = x.shape[1]

early_stop = tf.keras.callbacks.EarlyStopping(patience=100, monitor='val_acc')

#%%
window_size = 20
x_rnn, y_rnn = create_multivariate_rnn_data(x, y, window_size=window_size)
x_rnn_train = x_rnn[:split]
x_rnn_test = x_rnn[split2:]
y_rnn_train = y_rnn[:split]
y_rnn_test = y_rnn[split2:]


#%%
# base.save('h5/base_data3_minmax_m5.h5')
base2.save('h5/base2_data3_minmax_D1.h5')
# base3.save('h5/base3_data3_minmax_m5.h5')

#%%
base = keras.Sequential([
    keras.layers.Dense(512, kernel_regularizer=keras.regularizers.l2(0.0001), activation=tf.nn.relu, input_shape=(n_cols, )),
    keras.layers.Dropout(0.75),
    keras.layers.Dense(512, kernel_regularizer=keras.regularizers.l2(0.0001), activation=tf.nn.relu),
    keras.layers.Dropout(0.75),
    keras.layers.Dense(256, kernel_regularizer=keras.regularizers.l2(0.0001), activation=tf.nn.relu),
    keras.layers.Dropout(0.75),
    keras.layers.Dense(1, activation=tf.nn.sigmoid)
])
base.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', 'binary_crossentropy', 'binary_accuracy'])
base_history = base.fit(x_train, y_train, epochs=200, callbacks=[early_stop], validation_split=0.3, validation_data=(x_test, y_test), batch_size=1024, verbose=2)

#%%
base2 = keras.Sequential([
    keras.layers.Dense(512, kernel_regularizer=keras.regularizers.l2(0.0001), activation=tf.nn.relu),
    keras.layers.Dense(64, kernel_regularizer=keras.regularizers.l2(0.0001), activation=tf.nn.relu, input_shape=(n_cols, )),
    keras.layers.Dropout(0.75),
    keras.layers.Dense(64, kernel_regularizer=keras.regularizers.l2(0.0001), activation=tf.nn.relu),
    keras.layers.Dense(1, activation=tf.nn.sigmoid)
])
base2.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', 'binary_crossentropy', 'binary_accuracy'])
base_history2 = base2.fit(x_train, y_train, epochs=200, callbacks=[early_stop], validation_split=0.3, validation_data=(x_test, y_test), batch_size=1024, verbose=2)


#%%
base3 = keras.Sequential([
    keras.layers.Dense(512, kernel_regularizer=keras.regularizers.l2(0.0001), activation=tf.nn.relu),
    keras.layers.Dense(64, kernel_regularizer=keras.regularizers.l2(0.0001), activation=tf.nn.relu, input_shape=(n_cols, )),
    keras.layers.Dropout(0.75),
    keras.layers.Dense(64, kernel_regularizer=keras.regularizers.l2(0.0001), activation=tf.nn.relu),
    keras.layers.Dropout(0.75),
    keras.layers.Dense(1, activation=tf.nn.sigmoid)
])
base3.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', 'binary_crossentropy', 'binary_accuracy'])
base_history3 = base3.fit(x_train, y_train, epochs=200, callbacks=[early_stop], validation_split=0.3, validation_data=(x_test, y_test), batch_size=1024, verbose=2)


#%%
plot_history([('base', base_history), ('base2', base_history2), ('base3', base_history3)])

#%%
print('base: ', base.evaluate(x_test, y_test))
print('base2: ', base2.evaluate(x_test, y_test))
print('base3: ', base3.evaluate(x_test, y_test))

#%%
lstm1 = keras.Sequential([
    keras.layers.LSTM(12, dropout=.2, recurrent_dropout=.2, input_shape=(x_rnn_train.shape[1], x_rnn_train.shape[2]), return_sequences=True),
    keras.layers.LSTM(6, dropout=.2, recurrent_dropout=.2),
    keras.layers.Dense(10, kernel_regularizer=keras.regularizers.l2(0.002), activation=tf.nn.relu),
    keras.layers.Dropout(0.25),
    keras.layers.Dense(1, activation=tf.nn.sigmoid)
])
lstm1.compile(optimizer='RMSProp', loss='binary_crossentropy', metrics=['accuracy', 'binary_crossentropy'])
lstm1_history = lstm1.fit(x_rnn_train, y_rnn_train, callbacks=[early_stop], validation_split=0.3, epochs=100, batch_size=1024, validation_data=(x_rnn_test, y_rnn_test), verbose=2)


#%%
lstm2 = keras.Sequential([
    keras.layers.LSTM(12, dropout=.2, recurrent_dropout=.2, input_shape=(x_rnn_train.shape[1], x_rnn_train.shape[2]), return_sequences=True),
    keras.layers.LSTM(12, dropout=.2, recurrent_dropout=.2),
    keras.layers.Dense(10, kernel_regularizer=keras.regularizers.l2(0.002), activation=tf.nn.relu),
    keras.layers.Dropout(0.75),
    keras.layers.Dense(1, activation=tf.nn.sigmoid)
])
lstm2.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', 'binary_crossentropy'])
lstm2_history = lstm2.fit(x_rnn_train, y_rnn_train, callbacks=[early_stop], validation_split=0.3, epochs=100, batch_size=1024, validation_data=(x_rnn_test, y_rnn_test), verbose=2)

#%%
lstm3 = keras.Sequential([
    keras.layers.LSTM(400, dropout=.2, recurrent_dropout=.2, input_shape=(x_rnn_train.shape[1], x_rnn_train.shape[2]), return_sequences=True),
    keras.layers.LSTM(400, dropout=.2, recurrent_dropout=.2),
    keras.layers.Dense(400, kernel_regularizer=keras.regularizers.l2(0.00001), activation=tf.nn.relu),
    keras.layers.Dense(1, activation=tf.nn.sigmoid)
])
lstm3.compile(optimizer='RMSProp', loss='binary_crossentropy', metrics=['accuracy', 'binary_crossentropy'])
lstm3_history = lstm3.fit(x_rnn_train, y_rnn_train, callbacks=[early_stop], validation_split=0.3, epochs=200, batch_size=1024, validation_data=(x_rnn_test, y_rnn_test), verbose=2)

#%%
plot_history([('lstm3', lstm3_history)])
print('lstm3: ', lstm3.evaluate(x_rnn_test, y_rnn_test))

#%%
plot_history([('lstm1', lstm1_history), ('lstm2', lstm2_history), ('lstm3', lstm3_history)])

#%%
plot_history([('base', base_history), ('base2', base_history2), ('3', base_history3), ('lstm1', lstm1_history), ('lstm2', lstm2_history)])


#%%
print('lstm1: ', lstm1.evaluate(x_rnn_test, y_rnn_test))
print('lstm2: ', lstm2.evaluate(x_rnn_test, y_rnn_test))
print('lstm3: ', lstm3.evaluate(x_rnn_test, y_rnn_test))