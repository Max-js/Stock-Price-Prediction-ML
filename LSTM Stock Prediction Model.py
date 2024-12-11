import yfinance as yf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from datetime import datetime
from dateutil.relativedelta import relativedelta

print("Welcome to the Northwest Investment stock analysis tool!")

ticker = input("Please input a ticker to analyze: ").upper()

current_datetime = datetime.now()

def getRollingEndDate():
    return f"{current_datetime.year}-{current_datetime.month}-{current_datetime.day+1}"

def getRollingStartDate():
    start_time = current_datetime - relativedelta(years=1)
    return f"{start_time.year}-{start_time.month}-{start_time.day}"

data = yf.download(ticker, start=getRollingStartDate(), end=getRollingEndDate())
data = data[['Close']]

#Normalize the data.
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data.to_numpy())

#Use last 90 days to predict the next day
length_of_historical_data = 60

x = []
y = []
for i in range(length_of_historical_data, len(scaled_data)):
    x.append(scaled_data[i - length_of_historical_data:i])
    y.append(scaled_data[i, 0])

x = np.array(x)
y = np.array(y)
x = np.reshape(x, (x.shape[0], x.shape[1], 1))

#Split data into train / test sets
train_size = int(len(x) * 0.8)
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

model.compile(optimizer='adam', loss='mean_squared_error')
fit_data = model.fit(x_train, y_train, epochs=20, batch_size=32, validation_data=(x_test, y_test))

predicted_prices = model.predict(x_test)
predicted_prices = scaler.inverse_transform(predicted_prices.reshape(-1, 1))
actual_prices = scaler.inverse_transform(y_test.reshape(-1, 1))

buy_signals = []
for i in range(1, len(predicted_prices)):
    #Buy if the predicted price is higher than the actual price
    if predicted_prices[i] > actual_prices[i]:
        buy_signals.append((data.index[-len(y_test)+i], actual_prices[i], predicted_prices[i]))

plt.figure(figsize=(12, 7))
plot_dates = np.array(data.index[-len(y_test):])
plt.plot(plot_dates, actual_prices, color='blue', label="Actual Prices", linewidth=2)
plt.plot(plot_dates, predicted_prices, color='red', label="Predicted Prices", linestyle='--')

labeled = False
for signal in buy_signals:
    if labeled == False:
        plt.scatter(signal[0], signal[1], color='green', label="Buy Signal", marker='^', alpha=0.7)
        labeled = True
    else:
        plt.scatter(signal[0], signal[1], color='green', marker='^', alpha=0.7)

plt.title(f'{ticker} Stock Price Prediction', fontsize=16)
plt.xlabel('Date', fontsize=12)
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=2))
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
plt.xticks(rotation=45)
plt.ylabel('Price', fontsize=12)
plt.legend(fontsize=10)
plt.grid(alpha=0.3)
plt.show()
                      