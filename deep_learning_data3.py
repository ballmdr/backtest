
#%%
def make_model(dense_layers, activation, dropout):
    '''Creates a multi-layer perceptron model
    
    dense_layers: List of layer sizes; one number per layer
    '''

    model = Sequential()
    for i, layer_size in enumerate(dense_layers, 1):
        if i == 1:
            model.add(Dense(layer_size, input_dim=X.shape[1]))
            model.add(Activation(activation))
        else:
            model.add(Dense(layer_size))
            model.add(Activation(activation))
    model.add(Dropout(dropout))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='Adam',
                  metrics=['binary_accuracy', auc_roc])

    return model

def auc_roc(y_true, y_pred):
    # any tensorflow metric
    value, update_op = tf.metrics.auc(y_true, y_pred)

    # find all variables created for this metric
    metric_vars = [i for i in tf.local_variables() if 'auc_roc' in i.name.split('/')[1]]

    # Add metric variables to GLOBAL_VARIABLES collection.
    # They will be initialized for new session.
    for v in metric_vars:
        tf.add_to_collection(tf.GraphKeys.GLOBAL_VARIABLES, v)

    # force to update metric values
    with tf.control_dependencies([update_op]):
        value = tf.identity(value)
        return value

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
df = pd.read_csv('data/data3/data3_M1.csv', parse_dates=['Datetime'], index_col='Datetime')



#%%
X = df.drop('Target', axis=1).values
y = df.Target.values

split = int(len(df) * 0.60)
X_train = X[:split]
X_tmp = X[split:]
y_train = y[:split]
y_tmp = y[split:]

half_split = int(len(X_tmp)*0.50)
X_test = X_tmp[:half_split]
X_final = X_tmp[half_split:]
y_test = y_tmp[:half_split]
y_final = y_tmp[half_split:]

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.40, random_state=42)


#%%
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


#%%
minmax = MinMaxScaler()
X_train = minmax.fit_transform(X_train)
X_tmp = minmax.transform(X_tmp)

#%%
param_grid = {'dense_layers': [[32], [32, 32], [64], [64, 64], [64, 64, 32], [64, 32], [128]],
              'activation'  : ['relu', 'tanh'],
              'dropout'     : [.25, .5, .75],
              }

clf = KerasClassifier(make_model, verbose=True)
grid_cv = GridSearchCV(clf, param_grid, cv=3, n_jobs=-1, verbose=True)
early_stop = EarlyStopping(monitor='auc_roc', patience=10, verbose=1)

grid_cv.fit(X_train, y_train, callbacks=[early_stop], verbose=True, epohs=50)
print('\nBest Score: {:.2%}'.format(grid_cv.best_score_))
print('\nBest Params:\n', pd.Series(grid_cv.best_params_))