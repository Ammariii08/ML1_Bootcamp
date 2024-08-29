# Import required Libraries

import streamlit as st
import pandas as pd
import ccxt
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Initialize Binance API (to fetch the data)

exchange = ccxt.binance()

#********************************************** APP CREATION STARTS HERE ***************************************************#

app_name = "Crypto Forecasting App"
st.title(app_name)
st.subheader('Forecast the price of BTC and ETH')

# Add an image from an online resource

st.image('https://images.moneycontrol.com/static-mcnews/2022/03/shutterstock_1446648077.jpg?impolicy=website&width=1600&height=900')

# Take input from user about start and end time

st.sidebar.header('Select the parameters from below: ')
start_date = st.sidebar.date_input('Start date', datetime(2024, 8, 27))
end_date = datetime.now()

# Convert date to Unix timestamps for Binance API

start_timestamp = int(datetime.timestamp(datetime.combine(start_date, datetime.min.time())) * 1000)
end_timestamp = int(datetime.timestamp(end_date) * 1000)

# Add ticker symbol list

tickers = {'BTC-USD': 'BTC/USDT', 'ETH-USD': 'ETH/USDT'}
selected_ticker = st.sidebar.selectbox('Select a cryptocurrency', list(tickers.keys()))

# Fetch data from Binance API

bars = exchange.fetch_ohlcv(tickers[selected_ticker], timeframe='1m', since=start_timestamp)
data = pd.DataFrame(bars, columns=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
data['Date'] = pd.to_datetime(data['Timestamp'], unit='ms')
data.set_index('Date', inplace=True)
data.drop(['Timestamp'], axis=1, inplace=True)
st.write(data)

# Plot the data

st.header('Data Visualization')
st.subheader('Plot of the Data')
st.write('**Note:** Select your specific date range on the side-bar, or zoom in on the plot and select your specific column')
fig = px.line(data.reset_index(), x='Date', y=data.columns, title=f'{data.columns} price of {selected_ticker}', width=1000, height=600)
st.plotly_chart(fig)

# Add a select box to select the column from data

column = st.selectbox('Select the column to be used for forecasting', data.columns)

# Subsetting the data

data = data[[column]]
st.write('Selected Data')
st.write(data)

# ADF Test to Check Stationarity

st.header('Is data Stationary?')
st.write(adfuller(data[column])[1] < 0.05)

# Decompose the data

st.header('Decompose the Data')
decomposition = seasonal_decompose(data[column], model='additive', period=12)
st.write(decomposition.plot())
st.write('## Plotting the decomposition in Plotly')
st.plotly_chart(px.line(x=data.index, y=decomposition.trend, title='Trend', width=1000, height=400).update_traces(line_color='Blue'))
st.plotly_chart(px.line(x=data.index, y=decomposition.seasonal, title='Seasonality', width=1000, height=400).update_traces(line_color='green'))
st.plotly_chart(px.line(x=data.index, y=decomposition.resid, title='Residuals', width=1000, height=400).update_traces(line_color='red', line_dash='dot'))

# Run the Model

p = st.slider('Select the value of p', 0, 12, 2)
d = st.slider('Select the value of d', 0, 5, 1)
q = st.slider('Select the value of q', 0, 5, 2)
seasonal_order = st.number_input('Select the value of seasonal order (s)', 0, 24, 12)  # Period for seasonality

model = SARIMAX(data[column], order=(p, d, q), seasonal_order=(p, d, q, seasonal_order))
model = model.fit()

# Print model summary

st.header('Model Summary')
st.write(model.summary())
st.write('-------------')

# Predict the future values for the next 60 minutes

st.write("<p style='color:green; font-size:50px; font-weight:bold;'> Forecasting the data</p>", unsafe_allow_html=True)
predictions = model.get_forecast(steps=60)  # Predict for the next 60 minutes
predicted_mean = predictions.predicted_mean

# Debug: Check the length of predictions

st.write(f"Number of predictions: {len(predicted_mean)}")

# Create a DataFrame for predicted values

try:
    if len(predicted_mean) == 60:
        # Create DataFrame for predictions
        predictions_df = pd.DataFrame({'Predicted': predicted_mean})
        # Set the index to the correct date range
        predictions_df.index = pd.date_range(start=data.index[-1] + timedelta(minutes=1), periods=60, freq='T')
        # Add Date column
        predictions_df.insert(0, 'Date', predictions_df.index)
        predictions_df.reset_index(drop=True, inplace=True)
        
        st.write('## Predictions', predictions_df)
    else:
        st.write("Warning: The number of predictions is not 60. Please check the model configuration.")
except Exception as e:
    st.write(f"An error occurred: {e}")

# Plot the data

fig = go.Figure()
# Add actual data to the plot
fig.add_trace(go.Scatter(x=data.index, y=data[column], mode='lines', name='Actual', line=dict(color='blue')))

# Add predicted data to the plot
if len(predicted_mean) == 60:
    # Ensure 'Date' column exists in predictions_df
    if 'Date' in predictions_df.columns:
        fig.add_trace(go.Scatter(x=predictions_df['Date'], y=predictions_df['Predicted'], mode='lines', name='Predicted', line=dict(color='red')))
    else:
        st.write("Error: 'Date' column missing in predictions_df.")
        
# Set title and axis labels
fig.update_layout(title='Actual VS Predicted', xaxis_title='Date', yaxis_title='Price', width=1000, height=400)

# Display the plot
st.plotly_chart(fig)


# TO RUN THIS APP: In terminal write command (streamlit run Forecasting_app.py)
