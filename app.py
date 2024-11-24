import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data
import yfinance as yf
from keras.models import load_model
import streamlit as st
from sklearn.preprocessing import MinMaxScaler

# Set the start and end dates
start = '2010-01-01'
end = '2019-12-31'

# Create a Streamlit title
st.title('Stock Trend Prediction')

# Get the user input for the stock ticker
user_input = st.text_input('Enter Stock Ticker', 'AAPL')

# Download the stock data
df = yf.download(user_input, start=start, end=end)

# Check if the DataFrame is empty
if df.empty:
    st.error("The DataFrame is empty. Please check your stock ticker or the date range.")
else:
    st.success("Data loaded successfully!")

    # Display the first few rows of the DataFrame
    st.write("Data from 2010 - 2024", df.head())

    # Check if the 'Close' column exists
    if 'Close' not in df.columns:
        st.error("'Close' column is not found in the DataFrame.")
    else:
        # Create training and testing datasets
        data_training = pd.DataFrame(df['Close'][0:int(len(df) * 0.70)])
        data_testing = pd.DataFrame(df['Close'][int(len(df) * 0.70):int(len(df))])

        # Check the shape and head of data_training
        st.write("Data training shape:", data_training.shape)
        st.write("Data training contents:\n", data_training.head())

        # If the DataFrame is defined, check for NaN values
        if data_training.isnull().values.any():
            st.warning("NaN values found in data_training. Dropping NaNs.")
            data_training = data_training.dropna()

        # Ensure it's 2D for the scaler
        if data_training.ndim == 1:
            data_training = data_training.values.reshape(-1, 1)

        # Print shape after processing
        st.write("Data training shape after processing:", data_training.shape)

        # Scaling the data
        scaler = MinMaxScaler(feature_range=(0, 1))
        data_training_array = scaler.fit_transform(data_training)

        # Describe the data
        st.subheader('Data from 2010 - 2024')
        st.write(df.describe())

        # Visualization
        st.subheader('Closing Price vs Time Chart')
        ma100 = df.Close.rolling(100).mean()
        fig = plt.figure(figsize=(12, 6))
        plt.plot(ma100, label='100-Day Moving Average', color='orange')
        plt.plot(df.Close, label='Closing Price', color='blue')
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.title('Closing Price vs Time')
        plt.legend()
        st.pyplot(fig)

        st.subheader('Closing Price vs Time Chart with 100MA')
        ma100 = df.Close.rolling(100).mean()
        fig = plt.figure(figsize=(12, 6))
        plt.plot(ma100, label='100-Day Moving Average', color='orange')
        plt.plot(df.Close, label='Closing Price', color='blue')
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.title('Closing Price vs Time with 100MA')
        plt.legend()
        st.pyplot(fig)

        st.subheader('Closing Price vs Time Chart with 100MA & 200MA')
        ma100 = df.Close.rolling(100).mean()
        ma200 = df.Close.rolling(200).mean()
        fig = plt.figure(figsize=(12, 6))
        plt.plot(ma100, label='100-Day Moving Average', color='orange')
        plt.plot(ma200, label='200-Day Moving Average', color='red')
        plt.plot(df.Close, label='Closing Price', color='blue')
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.title('Closing Price vs Time with 100MA & 200MA')
        plt.legend()
        st.pyplot(fig)

        # Load the model
        try:
            model = load_model('keras_model.keras')
        except ValueError:
            st.error("Model file not found. Please ensure 'keras_model.keras' is in the same directory as this script.")

        # Testing part
        past_100_days = data_training.tail(100)
        final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
        input_data = scaler.transform(final_df)

        # Initialize x_test and y_test as lists
        