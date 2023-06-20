import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import math


def plot(test, predicted, company):
    plt.style.use('ggplot')
    plt.clf()
    plt.plot(test, color='red', label=f'{company} Close Price')
    plt.plot(predicted, color='blue', label=f'Predicted {company} Close Price')
    plt.title(f'{company} Close Price Prediction')
    plt.xlabel('Time')
    plt.ylabel(f'{company} Close Price')
    plt.legend()
    company = company.lower()
    plt.savefig(f'figures/{company}_prediction.png')


def root_mean_squared_error(test, predicted):
    mse = mean_squared_error(test, predicted)
    rmse = math.sqrt(mse)
    return rmse
