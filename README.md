# ML Stock Prediction 

This repository contains a machine learning-based stock prediction project created by Josh Puglielli. The project aims to predict the future prices of stocks using historical stock market data and a sequential recurrent neural network (RNN) model. 


## Introduction 

The ML Stock Prediction project utilizes machine learning techniques to predict the future prices of stocks based on historical data. The prediction models are trained on a dataset containing historical stock market information, including opening price, closing price, highest price, lowest price, and trading volume. By analyzing these features, the models can make predictions about future stock prices, providing potential insights for investors and traders.

## Installation 

To use this project, follow these steps:

1. Clone the repository:

    ```bash
    git clone https://github.com/JoshPugli/ml-stock-prediction.git 
    ```

2. Install the required dependencies. It is recommended to use a virtual environment:

    ```bash
    pip install -r requirements.txt 
    ```

3. Once the dependencies are installed, you can proceed to use the project.

## Usage

Run the pred.py script with the desired stock name as the command-line argument. The list of available stocks can be found in the /data/ folder of the repository. For example, to predict the stock prices for Apple (AAPL):

```bash
python pred.py AAPL
```
This command will use the pred.py script to perform stock price prediction specifically for the Apple (AAPL) stock.

Feel free to replace AAPL with any other stock name available in the /data/ folder of the repository. Remember to adjust the command accordingly based on your specific needs.


## Data 

The historical stock market data used in this project is included in the repository's data folder. The dataset contains a collection of historical stock market information, including opening price, closing price, highest price, lowest price, and trading volume for various stocks.

You can find the data files in the following location within the repository:

```bash
/data/
```

The script provided in this project is designed to read and process the historical stock market data from the data files, allowing you to train the prediction models and make future price predictions based on the historical information.

## Models 

This project utilizes a Long Short-Term Memory (LSTM) model for stock price prediction. The LSTM model is implemented using the Keras library.

The architecture of the LSTM model is as follows:

```python
model = Sequential()

model.add(LSTM(units=units, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(Dropout(dropout))
model.add(LSTM(units=units, return_sequences=True))
model.add(Dropout(dropout))
model.add(LSTM(units=units, return_sequences=True))
model.add(Dropout(dropout))
model.add(LSTM(units=units))
model.add(Dropout(dropout))
model.add(Dense(units=1))
```

The LSTM model used for stock price prediction includes the following components:

- LSTM layers are stacked one after another to capture temporal dependencies in the stock price data.
- Dropout layers are added after each LSTM layer to prevent overfitting.
- The number of LSTM units/neurons in each layer is controlled by the `units` parameter.
- The final dense layer with a single unit is responsible for predicting the target variable, which is the stock price.

The model is compiled using the RMSprop optimizer and the mean squared error (MSE) loss function. It is trained on the training data using the `fit()` method.

Once trained, the model is used to make predictions on the test data. The predicted stock prices are then inverse-transformed using a scaler to obtain the actual stock prices. Finally, the `plot()` function visualizes the actual and predicted stock prices, and the `root_mean_squared_error()` function calculates the root mean squared error between the predicted and actual prices.

This LSTM model is designed to capture patterns and dependencies in the historical stock market data, enabling it to make predictions about future stock prices.


## Results 

<div style="display:flex;">
  <img src="/figures/amazon_prediction.png" alt="Image 1" style="width:33%;">
  <img src="/figures/apple_prediction.png" alt="Image 2" style="width:33%;">
  <img src="/figures/google_prediction.png" alt="Image 3" style="width:33%;">
</div>

These are sample results from AMZN, AAPL, and GOOG stock prices. The Root Mean Squared error for each is &approx;40.
