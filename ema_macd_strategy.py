import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import datetime 
import yfinance as yf
import warnings

# some setting for better output looking 
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

end_date = datetime.datetime.now()
end_date = end_date.strftime('%Y-%m-%d')
start_date = pd.to_datetime(end_date) - pd.DateOffset(365 * 10)
ticker = 'AAPL'  # you can change the ticker to any other stock you want to analyze
df = yf.download(tickers=ticker, start=start_date, end=end_date)

# index settings
df.columns = ['Adj Close', 'Close', 'High', 'Low', 'Open', 'Volume']
df = df.reset_index()
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)


#calcualte some indicators
def calculate_bollinger_bands(series, window=20, num_std=2):
    sma = series.rolling(window=window).mean()
    rolling_std = series.rolling(window=window).std()
    bb_Low = sma - (num_std * rolling_std)
    bb_High = sma + (num_std * rolling_std)
    bb_mid = sma
    return bb_Low, bb_mid, bb_High
df['bb_Low'], df['bb_mid'], df['bb_High'] = calculate_bollinger_bands(df['Close'], window=20, num_std=2)
def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi
df['RSI'] = calculate_rsi(df['Close'], period=14)
def calculate_cci(df, period=20):
    tp = (df['High'] + df['Low'] + df['Close']) / 3
    sma = tp.rolling(window=period).mean()
    mad = tp.rolling(window=period, min_periods=1).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
    cci = (tp - sma) / (0.015 * mad)
    return cci
df['CCI'] = calculate_cci(df, period=20)
def calculate_cmo(df, period=14):
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    sum_gain = gain.rolling(window=period, min_periods=1).sum()
    sum_loss = loss.rolling(window=period, min_periods=1).sum()
    cmo = 100 * (sum_gain - sum_loss) / (sum_gain + sum_loss)
    return cmo
df['CMO'] = calculate_cmo(df, period=14)
def calculate_mfi(df, period=14):
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    money_fLow = typical_price * df['Volume']

    positive_fLow = money_fLow.where(typical_price > typical_price.shift(1), 0)
    negative_fLow = money_fLow.where(typical_price < typical_price.shift(1), 0)

    positive_mf = positive_fLow.rolling(window=period).sum()
    negative_mf = negative_fLow.rolling(window=period).sum()

    mfi = 100 - (100 / (1 + (positive_mf / negative_mf)))
    return mfi
df['MFI'] = calculate_mfi(df, period=14)
def calculate_momentum(df, period=14):
    momentum = df['Close'].diff(period)
    return momentum
df['momentum'] = calculate_momentum(df, period=14)
def calculate_atr(group, period=14):
    High_Low = group['High'] - group['Low']
    High_Close = np.abs(group['High'] - group['Close'].shift())
    Low_Close = np.abs(group['Low'] - group['Close'].shift())
    tr = pd.concat([High_Low, High_Close, Low_Close], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    return atr
df['atr'] = df.groupby(df.index.year, group_keys=False).apply(calculate_atr)
def calculate_macd(df, short_period=12, long_period=26, signal_period=9):
    short_ema = df['Close'].ewm(span=short_period, adjust=False).mean()
    long_ema = df['Close'].ewm(span=long_period, adjust=False).mean()
    macd = short_ema - long_ema
    signal = macd.ewm(span=signal_period, adjust=False).mean()
    return macd, signal
def calculate_bollinger_bands(series, window=20, num_std=2):
    sma = series.rolling(window=window).mean()
    rolling_std = series.rolling(window=window).std()
    bb_low = sma - (num_std * rolling_std)
    bb_high = sma + (num_std * rolling_std)
    bb_mid = sma
    return bb_low, bb_mid, bb_high
df['bb_low'], df['bb_mid'], df['bb_high'] = calculate_bollinger_bands(df['Close'], window=20, num_std=2)
df['MACD_b'], df['MACD_Signal_o'] = calculate_macd(df) 
df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
df['EMA_200'] = df['Close'].ewm(span=200, adjust=False).mean()

#calcualting buy and sell signals based on the ema50, ema200 and macd
def buy_signal(df):
    buy_dates = []

    condition = (df['EMA_50'] > df['EMA_200'])
    valid_periods = df.loc[condition]

    for i in range(1, len(valid_periods)):
        if valid_periods['MACD_b'].iloc[i - 1] <= valid_periods['MACD_Signal_o'].iloc[i - 1] and \
                valid_periods['MACD_b'].iloc[i] > valid_periods['MACD_Signal_o'].iloc[i]:
            buy_dates.append(valid_periods.index[i])

    return buy_dates

def sell_signal(df):
    sell_dates = []

    condition = (df['EMA_50'] < df['EMA_200'])
    below_periods = df.loc[condition]

    blocks = []
    current_block = []

    for i in range(len(below_periods)):
        if i > 0 and below_periods.index[i] != below_periods.index[i - 1] + pd.Timedelta(days=1):
            if current_block:
                blocks.append(current_block)
            current_block = []
        current_block.append(below_periods.index[i])
    if current_block:
        blocks.append(current_block)

    for block in blocks:
        block_df = df.loc[block[0]:block[-1]]
        macd_below = block_df.loc[block_df['MACD_b'] < block_df['MACD_Signal_o']]
        macd_cross_dates = macd_below.index[macd_below['MACD_b'].diff().gt(0)].tolist()
        sell_dates.extend(macd_cross_dates)

    return sell_dates
buy_dates = buy_signal(df)
sell_dates = sell_signal(df)
buy_signals = pd.to_datetime(buy_dates)
sell_signals = pd.to_datetime(sell_dates)
buy_df = pd.DataFrame({"Signal": buy_signals, "Type": "Buy"})
sell_df = pd.DataFrame({"Signal": sell_signals, "Type": "Sell"})
combined_df = pd.concat([buy_df, sell_df]).sort_values(by="Signal").reset_index(drop=True)
combined_df['Signal_Number'] = combined_df.groupby('Type').cumcount() + 1
combined_df['Signal_Name'] = combined_df.apply(
    lambda row: f"{row['Type']} Signal ({row['Signal_Number']})", axis=1
)

# calculate entry, target, stop loss, potential profit and potential loss
# and cocat them to the final dataframe
entry_prices = []
target_prices = []
stop_losses = []
potential_profits = []
potential_losses = []
potential_profits_percent = []
potential_losses_percent = []

for idx, row in combined_df.iterrows():
    signal_date = row['Signal']

    if signal_date in df.index:
        entry_price = df.loc[signal_date, 'Close']
        atr = df.loc[signal_date, 'atr']

        if row['Type'] == "Buy":
            target_price = entry_price + (3 * atr)
            stop_loss = entry_price - (2 * atr)
        else:  # Sell
            target_price = entry_price - (3 * atr)
            stop_loss = entry_price + (2 * atr)

        entry_prices.append(entry_price)
        target_prices.append(target_price)
        stop_losses.append(stop_loss)

        potential_profit = abs(target_price - entry_price)
        potential_loss = abs(entry_price - stop_loss)

        potential_profits.append(potential_profit)
        potential_losses.append(potential_loss)

        potential_profit_percent = (potential_profit / entry_price) * 100
        potential_loss_percent = (potential_loss / entry_price) * 100

        potential_profits_percent.append(f"+{potential_profit_percent:.2f}%")
        potential_losses_percent.append(f"-{potential_loss_percent:.2f}%")
    else:
        entry_prices.append(np.nan)
        target_prices.append(np.nan)
        stop_losses.append(np.nan)
        potential_profits.append(np.nan)
        potential_losses.append(np.nan)
        potential_profits_percent.append(np.nan)
        potential_losses_percent.append(np.nan)

combined_df['Entry_Price'] = entry_prices
combined_df['Target_Price'] = target_prices
combined_df['Stop_Loss'] = stop_losses
combined_df['Potential_Profit'] = potential_profits
combined_df['Potential_Loss'] = potential_losses
combined_df['Potential_Profit_%'] = potential_profits_percent
combined_df['Potential_Loss_%'] = potential_losses_percent

final_df = combined_df[['Signal_Name', 'Signal', 'Entry_Price', 'Target_Price', 'Stop_Loss',
                        'Potential_Profit', 'Potential_Loss', 'Potential_Profit_%', 'Potential_Loss_%']]
final_df.set_index('Signal_Name', inplace=True)

print(final_df)
