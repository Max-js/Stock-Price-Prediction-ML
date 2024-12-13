import yfinance as yf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import MaxNLocator
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

len_of_prediction_data = 60 #Determine how many previous days data to use to make predictions (y data)

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

#Number of iterations over the data - changes graph x axis scale to match.
epochs = 20
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

#Plot actual vs. predicted values, buy signals
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

#Plot training and validation loss
plt.figure(figsize=(10, 6))
plt.plot(fit_data.history['loss'], label='Training Loss', color='blue')
plt.plot(fit_data.history['val_loss'], label='Validation Loss', color='orange')
plt.title('Model Loss Over Epochs', fontsize=16)
plt.xlabel('Epochs', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.legend(fontsize=10)
plt.grid(alpha=0.3)
plt.xticks(range(0, epochs, 1))
plt.show()

#Visualize model performance metrics
metrics = ['MAE', 'RMSE', 'R-Squared']
values = [mean_absolute_err, root_mean_squared_err, r2]

plt.figure(figsize=(8, 6))
plt.bar(metrics, values, color=['blue', 'orange', 'green'], alpha=0.7)
plt.title('Model Performance Metrics', fontsize=16)
plt.ylabel('Value', fontsize=12)
for i, v in enumerate(values):
    plt.text(i, v + 0.01, f"{v:.4f}", ha='center', fontsize=12, color='black')

plt.grid(alpha=0.3)
plt.show()

#Print UI stuff based on results? Buy - yes or no? Prediction worthiness (r2 value decent?)? 
#Check that it meets guidelines
