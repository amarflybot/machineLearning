import numpy as np
import pandas as pd
from sklearn.svm import LinearSVC


def main():
    dataFile = '/home/amarendra/Downloads/ad.data'
    data = pd.read_csv(dataFile, sep=",", header=None, low_memory=False)
    # print data.head(100)
    training_data = data.iloc[0:, 0:-1].apply(_series_to_num)
    training_data = training_data.dropna()
    training_labels = data.iloc[training_data.index, -1].apply(_to_label)

    # Training phase
    clf = LinearSVC()
    clf.fit(training_data[100:2300], training_labels[100:2300])

    # Test Phase
    predict = clf.predict(training_data.iloc[12].values.reshape(1, -1))
    print(predict)
    # df = pd.DataFrame.from_csv('/home/amarendra/Downloads/ad.data', parse_dates=False)
    # df.b.plot(color='g', lw=1.3)
    # df.c.plot(color='r', lw=1.3)


# check whether a given value is missing value, if yes change it to NaN

def _to_num(cell):
    try:
        return np.float(cell)
    except:
        return np.nan


def _series_to_num(series):
    return series.apply(_to_num)


def _to_label(_str_):
    if _str_ == "ad.":
        return 1
    else:
        return 0


if __name__ == "__main__": main()
