
#%%
def return_plot(estimator, plot=False):
    df['Predicted_Signal'] = estimator.predict(X)
    df.Predicted_Signal[df.Predicted_Signal > 0.5] = 1
    df.Predicted_Signal[df.Predicted_Signal < 0.5] = -1
    Cumulative_returns = np.cumsum(df[split:]['Returns'])
    df['Startegy_returns'] = df['Returns']* df['Predicted_Signal'].shift(1)
    Cumulative_Strategy_returns = np.cumsum(df[split:]['Startegy_returns'])
    print('Return: {}%'.format(round(Cumulative_Strategy_returns[-1]*100, 2)))
    
    if plot:
        plt.figure(figsize=(10,5))
        plt.plot(Cumulative_returns, color='r',label = 'Returns')
        plt.plot(Cumulative_Strategy_returns, color='g', label = 'Strategy Returns')
        plt.legend()
        plt.show()

def accuracy_plot(estimator):
    # summarize history for accuracy
    plt.plot(estimator.history.history['acc'])
    plt.plot(estimator.history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(estimator.history.history['loss'])
    plt.plot(estimator.history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

def compare_plot(model_1, model_2):
    plt.plot(model_1.history.history['val_loss'], 'r', model_2.history.history['val_loss'], 'b')
    plt.xlabel('Epochs')
    plt.ylabel('Validation Score')
    plt.legend(['first', 'second'])
    plt.show()


#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

import keras
from keras.models import load_model
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional, Activation, TimeDistributed
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.optimizers import SGD, Adagrad, Adam
from keras.wrappers.scikit_learn import KerasClassifier

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix


#%%
df = pd.read_csv('data/data3_M1.csv', parse_dates=['Datetime'], index_col='Datetime')



#%%
X = df.drop('Target', axis=1).values
y = df.Target.values

# split = int(len(df) * 0.60)
# X_train = X[:split]
# X_test = X[split:]
# y_train = y[:split]
# y_test = y[split:]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.60, random_state=42)


#%%
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


#%%
early_stop = EarlyStopping(patience=5, monitor='acc')
sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
ada = Adagrad(lr=0.01, epsilon=None, decay=0.0)
adam = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)


#%%
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(X.shape[1],)))
model.add(Dense(64, activation='relu'))
model.add(Dropout(.5))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=30, batch_size=128, validation_split=0.3, callbacks=[early_stop], verbose=True)

model.save('keras.h5')
