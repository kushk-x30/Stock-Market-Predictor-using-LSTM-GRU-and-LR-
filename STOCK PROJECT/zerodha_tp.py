import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, LSTM, Dense, Dropout
from datetime import datetime
import requests
import json
import webbrowser

st.set_page_config(page_title="Stock Market Predictor & Trader", layout="centered")

if "page_name" not in st.session_state:
    st.session_state.page_name = "welcome"

st.image("D:\\FY-AIDS\\Python\\College Work\\PROJECT\\STOCK MARKET PREDICTOR.png", use_container_width=True)
if st.session_state.page_name == "welcome":
    st.title("👋 Good Day!")
    st.markdown("Welcome to the **Stock Market Predictor & Trader** 🔮")
    st.markdown("Enter a stock ticker (like `AAPL`, `MSFT`, `INFY.NS`) or upload your own CSV to get started.")

    ticker_input = st.text_input("Enter Ticker Symbol (Example: INFY.NS)", value="")
    file_upload = st.file_uploader("Or Upload Your CSV", type=['csv'])

    if ticker_input or file_upload:
        st.session_state.ticker_input = ticker_input
        st.session_state.file_upload = file_upload
        st.session_state.page_name = "predict"
        st.rerun()

elif st.session_state.page_name == "predict":
    ticker_input = st.session_state.ticker_input
    file_upload = st.session_state.file_upload

    st.title(f"📈 {ticker_input.upper()} Stock Analysis")
    st.markdown("<hr>", unsafe_allow_html=True)

    def load_data(ticker=None, file=None):
        if file is not None:
            df = pd.read_csv(file)
        else:
            df = yf.download(ticker, period="15y")
            df.reset_index(inplace=True)
        return df

    df = load_data(ticker=ticker_input, file=file_upload)

    if 'Date' not in df.columns or 'Close' not in df.columns:
        st.error("CSV or ticker data must contain 'Date' and 'Close' columns.")
        st.stop()

    df = df[['Date', 'Close']].dropna()
    df['Date'] = pd.to_datetime(df['Date'])
    df.sort_values('Date', inplace=True)
    df.reset_index(drop=True, inplace=True)

    if df.empty:
        st.error("Loaded dataset has no valid rows after cleaning.")
        st.stop()

    st.subheader("📊 Latest Data Preview")
    st.dataframe(df.tail(10))

    fig1, ax1 = plt.subplots(figsize=(12, 5))
    ax1.plot(df['Date'], df['Close'], label='Close Price', color='blue')
    ax1.set_title("Close Price Over Time")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Price (INR)")
    ax1.grid(True)
    ax1.legend()
    st.pyplot(fig1)

    df_scaled = df.copy()
    df_scaled.set_index('Date', inplace=True)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df_scaled[['Close']].values)
    look_back = 60

    def create_sequences(data, look_back):
        X, y = [], []
        for i in range(look_back, len(data)):
            X.append(data[i - look_back:i, 0])
            y.append(data[i, 0])
        X, y = np.array(X), np.array(y)
        return X.reshape(X.shape[0], X.shape[1], 1), y

    X_train, y_train = create_sequences(scaled_data, look_back)

    def build_lstm():
        model = Sequential([
            LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
            Dropout(0.2),
            LSTM(units=50),
            Dropout(0.2),
            Dense(25),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model

    def build_gru():
        model = Sequential([
            GRU(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
            Dropout(0.2),
            GRU(units=50),
            Dropout(0.2),
            Dense(25),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model

    lstm_model = build_lstm()
    gru_model = build_gru()

    with st.spinner('Training LSTM model...'):
        lstm_model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)

    with st.spinner('Training GRU model...'):
        gru_model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)

    test_data = scaled_data[-look_back:]
    X_test = np.array([test_data]).reshape((1, look_back, 1))

    pred_lstm = float(scaler.inverse_transform(lstm_model.predict(X_test))[0][0])
    pred_gru = float(scaler.inverse_transform(gru_model.predict(X_test))[0][0])

    df_lr = df.copy()
    df_lr['Day'] = np.arange(len(df_lr))
    X = df_lr[['Day']].values
    y = df_lr['Close'].values
    split_idx = int(len(X) * 0.8)
    X_train_lr, X_test_lr = X[:split_idx], X[split_idx:]
    y_train_lr, y_test_lr = y[:split_idx], y[split_idx:]
    scaler_lr = StandardScaler()
    X_train_scaled = scaler_lr.fit_transform(X_train_lr)
    X_test_scaled = scaler_lr.transform(X_test_lr)
    model_lr = LinearRegression()
    model_lr.fit(X_train_scaled, y_train_lr)

    next_day = np.array([[df_lr['Day'].iloc[-1] + 1]])
    next_day_scaled = scaler_lr.transform(next_day)
    pred_lr = float(model_lr.predict(next_day_scaled)[0])

    today_price = float(df['Close'].iloc[-1])

    def get_recommendation(current, predicted):
        if predicted > current * 1.01:
            return "📈 Buy"
        elif predicted < current * 0.99:
            return "📉 Sell"
        else:
            return "⏸️ Hold"

    rec_lstm = get_recommendation(today_price, pred_lstm)
    rec_gru = get_recommendation(today_price, pred_gru)
    rec_lr = get_recommendation(today_price, pred_lr)

    st.subheader("📌 Stock Price Prediction")
    st.metric("Today's Close", f"₹{today_price:.2f}")
    st.metric("LSTM Prediction", f"₹{pred_lstm:.2f} ({rec_lstm})")
    st.metric("GRU Prediction", f"₹{pred_gru:.2f} ({rec_gru})")
    st.metric("Linear Regression Prediction", f"₹{pred_lr:.2f} ({rec_lr})")

    st.markdown("---")
    st.subheader("💰 Real Stock Trading via Zerodha")

    st.markdown("To trade using real money, you must authenticate with your Zerodha Kite account.")
    kite_api_key = st.text_input("Enter your Zerodha API Key")
    kite_api_secret = st.text_input("Enter your Zerodha API Secret", type="password")

    if kite_api_key and kite_api_secret:
        login_url = f"https://kite.zerodha.com/connect/login?v=3&api_key={kite_api_key}"
        st.markdown(f"[🔐 Click here to Login via Zerodha]({login_url})")
        st.info("Once logged in, obtain the request token and generate access token on backend to complete trading integration.")

        request_token = st.text_input("Paste your request token from redirect URL")

        if request_token:
            st.code("Backend server required to exchange request token for access token using kiteconnect")


