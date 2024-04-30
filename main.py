import pandas as pd
import numpy as np
from neuralforecast import NeuralForecast
from neuralforecast.models import LSTM, NHITS, RNN
import yfinance as yf
from datetime import datetime

# Visualization Imports
import plotly.graph_objects as go
import plotly.io as pio


# Colored prints
class col:
    cyan = '\033[96m'
    blue = '\033[94m'
    green = '\033[92m'
    yellow = '\033[93m'
    red = '\033[91m'
    bold = '\033[1m'
    end = '\033[0m'
    
pio.templates.default = "plotly_dark"

symbol = 'BTC-USD'
start_date = '2023-01-01'
end_date = datetime.now()

Y_df = yf.download(symbol, start=start_date, end=end_date)

print(Y_df.head())

Y_df.index = pd.to_datetime(Y_df.index)

Y_df = Y_df.reset_index()

Y_df.columns = Y_df.columns.str.lower()

Y_df = Y_df.rename(columns={'close': 'y', 'date': 'ds'})

Y_df['unique_id'] = 1

horizon = 12

# Try different hyperparmeters to improve accuracy.
models = [LSTM(h=horizon,                    # Forecast horizon
               max_steps=500,                # Number of steps to train
               scaler_type='standard',       # Type of scaler to normalize data
               encoder_hidden_size=64,       # Defines the size of the hidden state of the LSTM
               decoder_hidden_size=64,),     # Defines the number of hidden units of each layer of the MLP decoder
          NHITS(h=horizon,                   # Forecast horizon
                input_size=2 * horizon,      # Length of input sequence
                max_steps=100,               # Number of steps to train
                n_freq_downsample=[2, 1, 1]) # Downsampling factors for each stack output
          ]

nf = NeuralForecast(models=models, freq='B')
nf.fit(df=Y_df)

Y_hat_df = nf.predict()
Y_hat_df = Y_hat_df.reset_index()
print(Y_hat_df.head())

Y_df_valid = Y_df.drop(Y_df.tail(12).index)
Y_df_test = Y_df.tail(12)

nf_valid = NeuralForecast(models=models, freq='B')
nf_valid.fit(df=Y_df_valid)

Y_hat_df_valid = nf_valid.predict()
Y_hat_df_valid = Y_hat_df_valid.reset_index()
print(Y_hat_df_valid.head())

# Error analysis
def MAE(values, predictions):
    errors = np.array(values) - np.array(predictions)
    return np.mean(abs(errors))

# Printing MAE values for each model
print(f"{col.bold+col.green +'NHITS MAE ='} {MAE(Y_df_test['y'], Y_hat_df_valid['NHITS'][:len(Y_df_test['y'])]):0.2f}")
print(f"{col.bold+col.yellow +'LSTM MAE ='} {MAE(Y_df_test['y'], Y_hat_df_valid['LSTM'][:len(Y_df_test['y'])]):0.2f}")

trace_lstm = go.Scatter(x=Y_hat_df['ds'], y=Y_hat_df['LSTM'], mode='lines', line=dict(color='red'), name='LSTM Pred')
trace_nhits = go.Scatter(x=Y_hat_df['ds'], y=Y_hat_df['NHITS'], mode='lines', line=dict(color='purple'), name='NHITS Pred')
trace_actual = go.Scatter(x=Y_df['ds'], y=Y_df['y'], mode='lines', line=dict(color='cyan'), name='Actual Close')
trace_lstm_valid = go.Scatter(x=Y_hat_df_valid['ds'], y=Y_hat_df_valid['LSTM'], mode='lines', line=dict(color='red'), name='LSTM Validation')
trace_nhits_valid = go.Scatter(x=Y_hat_df_valid['ds'], y=Y_hat_df_valid['NHITS'], mode='lines', line=dict(color='purple'), name='NHITS Validation')

# Defining layout
layout = go.Layout(
    title=F'{symbol} Prediction Comparison',
    xaxis=dict(title='Date'),
    yaxis=dict(title=f'{symbol} - Price'),
    showlegend=True,
    legend=dict(x=0.02, y=0.98)
)

# Creating figure
fig = go.Figure(data=[trace_lstm, trace_nhits, trace_actual, trace_lstm_valid, trace_nhits_valid], layout=layout)

# Set x and y-axis ranges
start_date = pd.to_datetime('2024-01-01')
end_date = pd.to_datetime('2024-05-31')
fig.update_xaxes(range=[start_date, end_date])
fig.show()