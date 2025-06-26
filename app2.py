import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import tensorflow as tf
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import datetime as dt

st.title('Stock Price Prediction using LSTM')
st.write("""This application predicts stock prices using an LSTM model. Please enter the stock ticker, start date, and end date to view predictions.""")

# User Inputs
today = dt.date.today()

stock = st.sidebar.text_input('Enter Stock Ticker (e.g., AAPL, MSFT):', 'NVDA')
start = st.sidebar.date_input('Start Date', value=pd.to_datetime('2000-01-01'))
end = st.sidebar.date_input('End Date', value=today)

def load_data(stock):
    data = yf.download(stock, start, end)
    data.reset_index(inplace=True)
    return data

if st.sidebar.button("Submit"):
    data = load_data(stock)

    MA_100 = data['Close'].rolling(window=100).mean()
    MA_200 = data['Close'].rolling(window=200).mean()
    ema100=data['Close'].ewm(span=100, adjust=False).mean()
    ema200=data['Close'].ewm(span=200, adjust=False).mean()

    st.subheader('Raw Data')
    st.write(data.tail())
    data.to_csv(f'{stock}_dataset.csv')
    df01=pd.read_csv(f'{stock}_dataset.csv')
    # Plot Data
    def plot_data():
        st.subheader("Original CandleStick Chart")
    
        figure=go.Figure(data=[go.Candlestick(x=df01["Date"], open=df01['Open'],high=df01['High'],
                                        low=df01['Low'], close=df01['Close'])])
        figure.update_layout(xaxis_rangeslider_visible=False)
        st.plotly_chart(figure)

    plot_data()

    def plot_line():
        # Plot
        st.subheader("Line chart")
        fig = px.line(df01, x='Date', y=['Close'])
        st.plotly_chart(fig)
    plot_line()

    def plot_ma_data():
        st.title("Moving Average Chart(100,200)")
        fig, ax = plt.subplots(figsize=(20,16))
        ax.plot(data['Date'], data['Close'], label='Close Price', color='blue',linewidth=2)
        ax.plot(data['Date'], MA_100, label='100-day MA', color='red', linestyle='dashed',linewidth=2)
        ax.plot(data['Date'], MA_200, label='200-day MA', color='green', linestyle='dashed',linewidth=2)
        ax.set_xlabel('Date')
        ax.set_ylabel('Close Price')
        ax.legend()
        st.pyplot(fig)

        st.subheader("Exponential Moving Average Chart(100,200)")
        fig1, ex=plt.subplots(figsize=(20,16))
        ex.plot(data['Date'], data['Close'], label='Close Price', color='blue',linewidth=2)
        ex.plot(data['Date'], ema100, label='100-day EMA', color='red', linestyle='dashed',linewidth=2)
        ex.plot(data['Date'], ema200, label='200-day EMA', color='green', linestyle='dashed',linewidth=2)
        ex.set_xlabel('Date')
        ex.set_ylabel('Close Price')
        ex.legend()
        st.pyplot(fig)

    plot_ma_data()

    data_training = pd.DataFrame(data['Close'][0:int(len(data)*0.70)])
    data_testing = pd.DataFrame(data['Close'][int(len(data)*0.70):int(len(data))])

    scaler = MinMaxScaler(feature_range=(0,1))

    data_training_array = scaler.fit_transform(data_training)

    x_train=[]
    y_train=[]

    for i in range(100, data_training_array.shape[0]):
        x_train.append(data_training_array[i-100:i])
        y_train.append(data_training_array[i,0])

    x_train,y_train=np.array(x_train), np.array(y_train)

    past_100_days = data_training.tail(100)
    final_df=pd.concat([past_100_days,data_testing], ignore_index=True)
    input_data = scaler.fit_transform(final_df)
        
    x_test, y_test = [], []
    for i in range(100, input_data.shape[0]):
        x_test.append(input_data[i - 100:i])
        y_test.append(input_data[i, 0])
    x_test, y_test = np.array(x_test), np.array(y_test)

    # Make predictions

    model = tf.keras.models.load_model('stock_lstm_model.h5')
    
    y_predicted = model.predict(x_test)
        
    # Inverse scaling for predictions
    scaler1 = scaler.scale_
    scale_factor = 1 / scaler1[0]
    y_predicted = y_predicted * scale_factor
    y_test = y_test * scale_factor

    
    st.title("LSTM Predicted Price vs Original Price")


    fig3, ax3 = plt.subplots(figsize=(20, 16))
    ax3.plot(y_test, 'g', label="Original Price", linewidth = 2)
    ax3.plot(y_predicted, 'r', label="Predicted Price", linewidth = 2)
    ax3.set_title("Prediction vs Original Trend")
    ax3.set_xlabel("Time")
    ax3.set_ylabel("Price")
    ax3.legend()

    st.pyplot(fig3)
    
    st.subheader("Next Day Preditiction")
    df01=df01.iloc[1:]
    df02=df01[['Close']].iloc[-5:]
    data_scaled3 = scaler.fit_transform(df02)
    data1=data_scaled3.reshape(1,5,1)
    pred1=model.predict(data1)
    predictons3 = scaler.inverse_transform(pred1)
    st.text(f'{predictons3[0,0]}')

    st.subheader("Next 5 Days Forecast using LSTM")

    future_predictions = []
    last_100_days = df01[['Close']].iloc[-100:]
    input_seq = scaler.fit_transform(last_100_days).reshape(1, 100, 1)

    for _ in range(5):
        pred = model.predict(input_seq)[0][0]
        future_predictions.append(pred)
        # Append predicted value to the input sequence
        input_seq = np.append(input_seq[:, 1:, :], [[[pred]]], axis=1)

    # Inverse transform the predictions
    future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1)).flatten()

    # Create future date range
    last_date = pd.to_datetime(df01['Date'].iloc[-1])
    future_dates = [last_date + pd.Timedelta(days=i) for i in range(1, 6)]

    # Plot the next 5-day predictions
    fig_future, ax_future = plt.subplots(figsize=(12, 6))
    ax_future.plot(future_dates, future_predictions, marker='o', color='orange', label='Predicted Close Price')
    ax_future.set_title("Next 5 Days Stock Price Forecast")
    ax_future.set_xlabel("Date")
    ax_future.set_ylabel("Predicted Close Price")
    ax_future.legend()
    st.pyplot(fig_future)
