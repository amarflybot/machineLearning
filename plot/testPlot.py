import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib import style


def main():
    style.use('ggplot')
    ts = pd.Series(np.random.randn(1000), index=pd.date_range('1/1/2000', periods=1000))
    ts = ts.cumsum()
    ts.plot()
    df = pd.DataFrame(np.random.randn(1000, 4), index=ts.index, columns=list('ABCD'))
    df = df.cumsum()
    # plt.figure()
    df.plot()
    plt.show()


if __name__ == "__main__": main()
