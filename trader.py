import numpy as np
import pandas as pd

from tensorflow import keras
from keras.models import Input, Model, Sequential, load_model
from keras.layers import Dense, Dropout, LSTM
from keras.callbacks import EarlyStopping, ModelCheckpoint

from sklearn.preprocessing import MinMaxScaler


scaler = MinMaxScaler(feature_range=(0, 1))


def load_data(path):
    df = pd.read_csv(path, names=['open', 'high', 'low', 'close'])
    x = scaler.fit_transform(df)
    return x


def create_X_Y(data: np.array, target: np.array, n_lag, n_ahead) -> tuple:
    X, Y = [], []
    for i in range(len(data) - n_lag - n_ahead):
        X.append(data[i:(i + n_lag)])
        Y.append(target[(i + n_lag):(i + n_lag + n_ahead)])
        
    X, Y = np.array(X), np.array(Y)
    X = np.reshape(X, (X.shape[0], n_lag, data.shape[1]))
    return X, Y


class Trader():
    def __init__(self, training_data):
        self.previous_data = training_data[-10:]
        self.slot = 0

        self.n_lag = 10
        self.n_ahead = 2
        self.n_ft = 4
        self.test_split = 0.95
        self.valid_split = 0.8
        self.epochs = 1000
        self.batch_size = 30

        d=0.3
        model= Sequential()
        model.add(LSTM(256, input_shape=(self.n_lag, self.n_ft), return_sequences=True))
        model.add(Dropout(d))
        model.add(LSTM(128, input_shape=(self.n_lag, self.n_ft), return_sequences=False))
        model.add(Dropout(d))
        model.add(Dense(16, activation='linear'))
        model.add(Dropout(d))
        model.add(Dense(self.n_ahead, activation='linear'))
        self.model = model

    def train(self, training_data): 
        empty_model = self.model
        empty_model.compile(loss='mse', optimizer='adam')

        x = training_data
        y = training_data[:,0]
        x, y = create_X_Y(data=x, target=y, n_lag=self.n_lag, n_ahead=self.n_ahead)
        x_, y_  = x[0:int(self.test_split*len(x))] , y[0:int(self.test_split*len(x))]
        x_test ,y_test = x[int(self.test_split*len(x)):] , y[int(self.test_split*len(x)):]
        x_train, y_train = x_[:int(self.valid_split*len(x_))] , y_[:int(self.valid_split*len(x_))]
        x_valid, y_valid = x_[int(self.valid_split*len(x_)):] , y_[int(self.valid_split*len(x_)):]

        callbacks_list = [
            EarlyStopping(patience=300, monitor = 'val_loss'),
            ModelCheckpoint('lstm.h5', monitor='val_loss', verbose=1, mode='min', save_best_only=True)
        ]
        history = empty_model.fit(
            x_train,
            y_train,
            batch_size=self.batch_size,
            shuffle=False,
            epochs=self.epochs,
            validation_data=(x_valid, y_valid),
            callbacks=callbacks_list
        )
        return history

    def predict_action(self, current_data):
        self.model = load_model('./lstm.h5')
        self.previous_data = np.concatenate((self.previous_data[-9:], current_data))
        pred = self.model.predict(self.previous_data.reshape(-1, 10, 4))
        tomorrow_price, future_price = pred[0][0], pred[0][1]
        # you already have one
        if self.slot == 1 and tomorrow_price > future_price:
            self.slot = 0
            return '-1'
        # you have nothing
        elif self.slot == 0:
            if tomorrow_price > future_price:
                self.slot = -1
                return '-1'
            if  tomorrow_price < future_price:
                self.slot = 1
                return '1'
        # you owe one
        elif self.slot == -1 and tomorrow_price < future_price:
            self.slot = 0
            return '1'
        return '0'

if __name__ == "__main__":
    # You should not modify this part.
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--training", default="training_data.csv", help="input training data file name")
    parser.add_argument("--testing", default="testing_data.csv", help="input testing data file name")
    parser.add_argument("--output", default="output.csv", help="output file name")
    args = parser.parse_args()

    # You can modify the following part at will.
    training_data = load_data(args.training)
    trader = Trader(training_data)
    # trader.train(training_data)

    testing_data = load_data(args.testing)
    with open(args.output, "w") as output_file:
        for row in testing_data[0:-1]:
            # We will perform your action as the open price in the next day.
            action = trader.predict_action(row.reshape(-1, 4))
            output_file.write(action + '\n')

            # this is your option, you can leave it empty.
            # trader.re_training()