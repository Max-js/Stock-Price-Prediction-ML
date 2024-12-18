import yfinance as yf
import numpy as np
import pandas as pd
import logging
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' #Suppress info and warning logs from Tensorflow
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

#Suppress pandas future warnings and yfinance error messages in favor of local error handling
pd.set_option('future.no_silent_downcasting', True)
logging.getLogger('yfinance').setLevel(logging.CRITICAL)

print("\033[1;32mWelcome to the Northwest Investment stock analysis tool!\033[0m")

current_datetime = datetime.now()

def get_rolling_end_date():
    tomorrow = current_datetime + timedelta(days=1)
    return tomorrow.strftime("%Y-%m-%d")

def get_rolling_start_date():
    start_time = current_datetime - relativedelta(years=2)
    return start_time.strftime("%Y-%m-%d")

def get_stock_data(ticker):
    try: 
        data = yf.download(ticker, start=get_rolling_start_date(), end=get_rolling_end_date())
        if data.empty:
            raise ValueError
        return data.dropna()
    except ValueError:
        print("\033[1;31mTicker not valid, please try again.\033[0m")
        return None

while True:
    ticker = input("\033[1;32mPlease input a ticker to analyze: \033[0m").upper()
    data = get_stock_data(ticker)
    
    if data is not None:
        data = data[['Close']]

        #Normalize the data.
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data.to_numpy())

        len_of_prediction_data = 100 #Determine how many previous days data to use to make predictions (y data)

        x = []
        y = []
        for i in range(len_of_prediction_data, len(scaled_data)):
            x.append(scaled_data[i - len_of_prediction_data:i])
            y.append(scaled_data[i, 0])

        x = np.array(x)
        y = np.array(y)
        x = np.reshape(x, (x.shape[0], x.shape[1], 1))

        #Split data into train / test sets
        len_of_historical_data = 0.7 #Determine how much historical data to use for training (x data)
        train_size = int(len(x) * len_of_historical_data)
        x_train = x[:train_size]
        x_test = x[train_size:]
        y_train = y[:train_size]
        y_test = y[train_size:]

        #Build LSTM Model
        model = Sequential([
            Input(shape=(x_train.shape[1], 1)),
            LSTM(units=50, return_sequences=True), #Adjust # of units to experiment. Less is faster, but could underfit. More is slower and could overfit.
            Dropout(0.2), #20% of nuerons are randomly ignored during training
            LSTM(units=50, return_sequences=False),
            Dropout(0.2), 
            Dense(units=1)
        ])

        epochs = 30 #Number of iterations over the data
        model.compile(optimizer='adam', loss='mean_squared_error')
        fit_data = model.fit(x_train, y_train, epochs=epochs, batch_size=32, validation_data=(x_test, y_test))

        #Make Predictions
        predicted_prices = model.predict(x_test)
        predicted_prices = scaler.inverse_transform(predicted_prices.reshape(-1, 1))
        actual_prices = scaler.inverse_transform(y_test.reshape(-1, 1))

        #Calculate metrics based on actual and predicted prices
        mean_absolute_err = mean_absolute_error(actual_prices, predicted_prices)
        root_mean_squared_err = np.sqrt(mean_squared_error(actual_prices, predicted_prices))
        r2 = r2_score(actual_prices, predicted_prices)

        buy_signals = []
        for i in range(1, len(predicted_prices)):
            #Buy if the predicted price is higher than the actual price
            if predicted_prices[i] > actual_prices[i]:
                buy_signals.append((data.index[-len(y_test)+i], actual_prices[i], predicted_prices[i]))

        #Determine whether stock is a good buy or not
        if buy_signals and buy_signals[-1][0]:
            latest_signal_date = buy_signals[-1][0].date()
            latest_predicted_price = buy_signals[-1][2][0]
            latest_actual_price = buy_signals[-1][1][0]
        else:
            latest_signal_date = None
            latest_predicted_price = 0
            latest_actual_price = 0

        if (latest_predicted_price > latest_actual_price) & (current_datetime.date() == latest_signal_date):
            message = f"{ticker} is potentially a good buy at this time."
            message_color = "green"
        else:
            message = f"{ticker} is most likely not a good buy at this time."
            message_color = "red"

        price_message = f"Current Price: {latest_actual_price:.2f} Predicted Price: {latest_predicted_price:.2f}"

        #Set up dashboard layout
        fig = plt.figure(figsize=(16, 12))
        gs = gridspec.GridSpec(2, 2, width_ratios=[2, 1], height_ratios=[1, 1], wspace=0.3, hspace=0.3)
        plot_dates = np.array(data.index[-len(y_test):])
        fig.text(0.5, 0.97, message, ha='center', va='center', fontsize=16, color=message_color, fontweight='bold')
        fig.text(0.5, 0.94, price_message, ha='center', va='center', fontsize=14, color="black", fontweight='bold')

        #Graph 1: Plot actual vs. predicted values, buy signals
        graph1 = fig.add_subplot(gs[0, 0])
        graph1.plot(plot_dates, actual_prices, color='blue', label="Actual Prices", linewidth=2)
        graph1.plot(plot_dates, predicted_prices, color='red', label="Predicted Prices", linestyle='--')

        labeled = False
        for signal in buy_signals:
            if labeled == False:
                graph1.scatter(signal[0], signal[1], color='green', label="Buy Signal", marker='^', alpha=0.7)
                labeled = True
            else:
                graph1.scatter(signal[0], signal[1], color='green', marker='^', alpha=0.7)

        graph1.set_title(f'{ticker} Stock Price Prediction', fontsize=16)
        graph1.set_xlabel('Date', fontsize=12)
        graph1.xaxis.set_major_locator(mdates.DayLocator(interval=7))
        graph1.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        graph1.tick_params(axis='x', rotation=45)
        graph1.set_ylabel('Price (USD)', fontsize=12)
        graph1.legend(fontsize=10)
        graph1.grid(alpha=0.3)

        #Graph 2: Plot training and validation loss
        graph2 = fig.add_subplot(gs[1, 0])
        graph2.plot(fit_data.history['loss'], label='Training Loss', color='blue')
        graph2.plot(fit_data.history['val_loss'], label='Validation Loss', color='orange')
        graph2.set_title('Model Loss Over Epochs', fontsize=16)
        graph2.set_xlabel('Epochs', fontsize=12)
        graph2.set_ylabel('Loss', fontsize=12)
        graph2.legend(fontsize=10)
        graph2.grid(alpha=0.3)
        graph2.set_xticks(range(0, epochs, 1))

        #Graph 3: Visualize model performance metrics
        metrics = ['MAE', 'RMSE', 'R-Squared']
        values = [mean_absolute_err, root_mean_squared_err, r2]

        graph3 = fig.add_subplot(gs[:, 1])
        graph3.bar(metrics, values, color=['blue', 'orange', 'green'], alpha=0.7)
        graph3.set_title('Model Performance Metrics', fontsize=16)
        graph3.set_ylabel('Value', fontsize=12)
        for i, v in enumerate(values):
            graph3.text(i, v + 0.01, f"{v:.4f}", ha='center', fontsize=12, color='black')
        graph3.grid(alpha=0.3)

        plt.show()
        break
