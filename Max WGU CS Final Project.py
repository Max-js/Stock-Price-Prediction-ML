import yfinance as yf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input

print("Welcome to the Northwest Investment stock analysis tool!")

ticker = input("Please input a ticker to analyze: ").upper()

#Should date range be input or hard coded?
data = yf.download(ticker, start="2024-01-01", end="2025-01-01")
data = data[['Close']]

#Normalize the data.
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data.to_numpy())

#Use last 90 days to predict the next day
length_of_historical_data = 90

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
model.fit(x_train, y_train, epochs=20, batch_size=32)

predicted_prices = model.predict(x_test)
predicted_prices = scaler.inverse_transform(predicted_prices.reshape(-1, 1))
actual_prices = scaler.inverse_transform(y_test.reshape(-1, 1))

#Gather buy signals for potential use in visual
buy_signals = []
for i in range(1, len(predicted_prices)):
    #Buy if the predicted price is higher than the actual price
    if predicted_prices[i] > actual_prices[i]:
        buy_signals.append((data.index[-len(y_test)+i], actual_prices[i], predicted_prices[i]))
        print(f"Buy signal on {data.index[-len(y_test)+i]}: Actual Price = {actual_prices[i]}, Predicted Price = {predicted_prices[i]}")

plot_dates = np.array(data.index[-len(y_test):])
plt.figure(figsize=(10, 6))
plt.plot(plot_dates, actual_prices, color='blue', label="Actual Prices")
plt.plot(plot_dates, predicted_prices, color='red', label="Predicted Prices")
plt.title(f'{ticker} Stock Price Prediction')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()
                      