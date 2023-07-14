from keras.layers import Dense, LSTM, Dropout
from keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import sys
import pandas as pd
from utils import plot, root_mean_squared_error


def get_data(name: str) -> pd.DataFrame:
    """
    Retrieve Amazon stock data from CSV file.

    Returns:
        pd.DataFrame: DataFrame with 'Close' prices, indexed by date.
    """
    try:
        df = pd.read_csv(f'./data/{name}_2006-01-01_to_2018-01-01.csv',
                        index_col="Date", parse_dates=True)
        df = df[["Close"]]  # Select only 'Close' price column
    except:
        print("Invalid company name. Please use a stock from the /data/ folder.")
        sys.exit(1)
    df.sort_values(by="Date", inplace=True)
    return df


def get_train_data(df: pd.DataFrame, window_size: int = 60) -> tuple:
    """
    Prepare training dataset from the given DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing Amazon stock data.
        window_size (int, optional): Number of data points to use for prediction. Defaults to 60.

    Returns:
        tuple: Training features and targets (X_train, y_train).
    """
    train = df[:'2016'].values  # Get data up to 2016
    # Define MinMaxScaler to scale data
    scaler = MinMaxScaler(feature_range=(0, 1))
    training_set_scaled = scaler.fit_transform(train)  # Scale the data

    X_train = []
    y_train = []

    for i in range(len(training_set_scaled) - window_size):
        X_train.append(training_set_scaled[i:i+window_size, 0])
        y_train.append(training_set_scaled[i+window_size, 0])

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    return X_train, y_train, scaler


def train_model(X_train: np.ndarray, y_train: np.ndarray, units: int = 50, dropout: float = 0.2, epochs: int = 25, batch_size: int = 64) -> Sequential:
    """
    Define and train LSTM model.

    Args:
        X_train (np.ndarray): Training features.
        y_train (np.ndarray): Training targets.
        units (int, optional): Number of LSTM units. Defaults to 50.
        dropout (float, optional): Dropout rate. Defaults to 0.2.
        epochs (int, optional): Number of training epochs. Defaults to 25.
        batch_size (int, optional): Batch size. Defaults to 64.

    Returns:
        Sequential: Trained LSTM model.
    """
    model = Sequential()

    model.add(LSTM(units=units, return_sequences=True,
              input_shape=(X_train.shape[1], 1)))
    model.add(Dropout(dropout))
    model.add(LSTM(units=units, return_sequences=True))
    model.add(Dropout(dropout))
    model.add(LSTM(units=units, return_sequences=True))
    model.add(Dropout(dropout))
    model.add(LSTM(units=units))
    model.add(Dropout(dropout))
    model.add(Dense(units=1))

    model.compile(optimizer='rmsprop', loss='mean_squared_error')

    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)

    return model


def get_test(df: pd.DataFrame, scaler, window_size: int = 60) -> tuple:
    """
    Prepare test dataset from the given DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing Amazon stock data.
        window_size (int, optional): Number of data points to use for prediction. Defaults to 60.

    Returns:
        tuple: Test features and actual values (X_test, test).
    """
    test = df['2017':].values
    dataset_total = pd.concat(
        (df["Close"][:'2016'], df["Close"]['2017':]), axis=0)

    inputs = dataset_total[len(dataset_total)-len(test) - window_size:].values
    inputs = scaler.transform(inputs.reshape(-1, 1))

    X_test = []
    for i in range(len(test)):
        X_test.append(inputs[i:i+window_size, 0])

    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    return X_test, test


# def main(company_name="Amazon"):
#     company_map = {"Amazon": "AMZN", "Google": "GOOGL", "Apple": "AAPL"}

#     df = get_data(company_map[company_name])
#     X_train, y_train, scaler = get_train_data(df)
#     X_test, test = get_test(df, scaler)

#     model = train_model(X_train, y_train)

#     predicted_stock_price = model.predict(X_test)
#     predicted_stock_price = scaler.inverse_transform(predicted_stock_price)

#     plot(test, predicted_stock_price, company_name)
#     print(root_mean_squared_error(test, predicted_stock_price))

# main()

arg1 = sys.argv[1]

df = get_data(arg1)
X_train, y_train, scaler = get_train_data(df)
X_test, test = get_test(df, scaler)

model = train_model(X_train, y_train)

predicted_stock_price = model.predict(X_test)
predicted_stock_price = scaler.inverse_transform(predicted_stock_price)

plot(test, predicted_stock_price, arg1)
print(root_mean_squared_error(test, predicted_stock_price))
