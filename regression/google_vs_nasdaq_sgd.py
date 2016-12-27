import pandas as pd
import numpy as np
from sklearn.linear_model import SGDRegressor


def main():
    googData = read_file('goog.csv')
    nasdaqData = read_file('nasdaq.csv')
    tbondData = read_file('tbond.csv', tbond=True)
    reg = SGDRegressor(eta0=0.1, n_iter=100000, fit_intercept=False)
    reg.fit((nasdaqData - tbondData).values.reshape(-1, 1), (googData - tbondData))
    print(reg.coef_)


def read_file(filename, tbond=False):
    data = pd.read_csv(filename, sep=",", usecols=[0, 6], names=['Date', 'Price'], header=0)
    if not tbond:
        returns = np.array(data["Price"][:-1], np.float) / np.array(data["Price"][1:], np.float) - 1
        data["Returns"] = np.append(returns, np.nan)
    if tbond:
        data["Returns"] = data["Price"] / 100
    data.index = data["Date"]
    data = data["Returns"][0:-1]
    return data


if __name__ == '__main__':
    main()
