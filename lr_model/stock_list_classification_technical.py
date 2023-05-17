# Import Packages
import datetime
import pandas as pd
import numpy as np
import yfinance as yf
import ta
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Ignore warnings
import warnings
warnings.filterwarnings('ignore')

# Start time
start_time = datetime.datetime.now()

# Input ticker code list
ticker_list = pd.read_csv('./data/stock_all_20230510.csv')['Ticker'].tolist()
ticker_list = [ticker + '.JK' for ticker in ticker_list]
print('Number of stocks in list: ', len(ticker_list), '\n')

# Select 'up' or 'next_up' as target
target_col = 'next_up'

# Feature parameters

# Input MA Cross Window
window_short_cross = 9
window_long_cross = 26
# Input EMA Window
window_ema = 20
# Input MACD Window
window_short_macd = 12
window_long_macd = 26
window_sign_macd = 9

feature_parameters = {
    'window_short_cross': window_short_cross,
    'window_long_cross': window_long_cross,
    'window_ema': window_ema,
    'window_short_macd': window_short_macd,
    'window_long_macd': window_long_macd,
    'window_sign_macd': window_sign_macd
}
pd.DataFrame.from_dict(feature_parameters, orient='index', columns=['Value']).to_csv('feature_parameters.csv')

# Blank dataframe/list to append data
data = pd.DataFrame(columns = ['Ticker', 'Training Data Size', 'Test Data Size', 'Accuracy', 'train_true', 'train_false', 'coef', 'intercept'])
no_data = []
model_error = []

for ticker in ticker_list:
    obj = yf.Ticker(ticker)
    df = obj.history(period='5y')
    if df.empty:
        print(obj, '\n')
        no_data.append(ticker)
    else:
        print(obj)
        
        # Features
        ema = ta.trend.EMAIndicator(close=df['Close'], window=window_ema).ema_indicator()
        sma_short = ta.trend.sma_indicator(close=df['Close'], window=window_short_cross, fillna=True)
        sma_long = ta.trend.sma_indicator(close=df['Close'], window=window_long_cross, fillna=True) 
        sma_cross = sma_short - sma_long
        sma_cross = sma_cross.rename(f'sma_cross_{window_short_cross}_{window_long_cross}')
        macd_line = ta.trend.MACD(df['Close'], window_slow=window_long_macd, window_fast=window_short_macd, window_sign=window_sign_macd).macd()
        macd_signal = ta.trend.MACD(df['Close'], window_slow=window_long_macd, window_fast=window_short_macd, window_sign=window_sign_macd).macd_signal()
        macd_diff = ta.trend.MACD(df['Close'], window_slow=window_long_macd, window_fast=window_short_macd, window_sign=window_sign_macd).macd_diff()
        rsi = ta.momentum.RSIIndicator(df['Close']).rsi()
        print('Feature calculated')
        feature = pd.concat([ema, sma_cross, macd_line, macd_signal, macd_diff, rsi], axis=1)
        feature = feature.pct_change().dropna()
        feature = feature.replace([np.inf, -np.inf], np.nan)
        feature = feature.dropna()
        
        # Target / Label
        df['up'] = False  # initialize 'up' column with 0
        df.loc[df['Close'] > df['Close'].shift(), 'up'] = True
        df['next_up'] = df['up'].shift(-1)
        target = df[df.index.isin(feature.index) == True][target_col].dropna().astype('bool')
        
        # Align Data Length Between Feature and Target
        feature = feature[feature.index.isin(target.index) == True]
        target = target[target.index.isin(feature.index) == True]
        
        # Classification : LogisticRegression
        test_size = int(0.2 * len(target))

        X_train, y_train = feature[:-test_size], target[:-test_size]
        X_test, y_test = feature[-test_size:], target[-test_size:]

        model = LogisticRegression()

        try:
            model.fit(X_train, y_train)
        except ValueError:
            print(f"ValueError occurred while fitting model for {ticker}. Skipping...", '\n')
            model_error.append(ticker)
            continue
        
        try:
            y_pred = model.predict(X_test)
        except ValueError:
            print(f"ValueError occurred while testing model for {ticker}. Skipping...", '\n')
            model_error.append(ticker)
            continue

        accuracy = accuracy_score(y_test, y_pred)
        
        print('Ticker:', ticker)
        print('Training data size: ', len(y_train))
        print('Test data size: ', len(y_test))
        print("Accuracy:", accuracy)
        print('\n')
        train_true = y_train.value_counts()[1]
        train_false = y_train.value_counts()[0]
        coef = model.coef_[0]
        intercept = model.intercept_[0]
        
        # Append to data
        new_row = {'Ticker': ticker, 'Training Data Size': len(y_train), 'Test Data Size': len(y_test), 'Accuracy': accuracy,'train_true':train_true, 'train_false':train_false, 'coef':coef, 'intercept':intercept}
        data = data.append(new_row, ignore_index=True)

# Print data info
data = data.set_index('Ticker')
print(data.info(), '\n')
data.to_csv('lr_model.csv')
# End time
end_time = datetime.datetime.now()

# Print Summary
print('Start time: ', start_time)
print('End time: ', end_time)
print('Time taken: ', end_time - start_time)
print('Feature parameters exported as feature_parameters.csv')
print('Model data exported as lr_model.csv')
print('Number of successfully analyzed stocks: ', len(data))
print('Number of data retrieval failure: ', len(no_data))
print('Number of model failure: ', len(model_error), '\n')
