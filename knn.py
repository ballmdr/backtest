import pickle
import pandas as pd
import backtrader as bt

with open('knn_data2_D1.pickle', 'rb') as file:
    model = pickle.load(file)


