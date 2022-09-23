import numpy as np


def random_noise(dataframe):

    return np.random.normal(scale=1.6, size=(len(dataframe),))

# b = pd.DataFrame({"sales": df_ordersnew2["GRANDTOTAL_NEW"].values[0:10],
#               "lag1": df_ordersnew2["GRANDTOTAL_NEW"].shift(1).values,
#               "lag2": df_ordersnew2["GRANDTOTAL_NEW"].shift(2).values,
#               "lag3": df_ordersnew2["GRANDTOTAL_NEW"].shift(3).values[0:10],
#               "lag4": df_ordersnew2["GRANDTOTAL_NEW"].shift(4).values[0:10]})

def roll_mean_features(dataframe, windows):
    for window in windows:
        dataframe['gtotal_roll_mean_' + str(window)] = dataframe["GRANDTOTAL_NEW"].transform(lambda x: x.shift(1).rolling(window=window,
                                                                                                       min_periods=2,
                                                                                                       win_type="triang").mean()) \
                                                      + random_noise(dataframe)


def lag_features(dataframe, lags):
    for lag in lags:
        dataframe['gtotal_lag_' + str(lag)] = dataframe["GRANDTOTAL_NEW"].transform(
            lambda x: x.shift(lag)) + random_noise(dataframe)
    return dataframe

def ewm_features(dataframe, alphas, lags):
    for alpha in alphas:
        for lag in lags:
            dataframe['sales_ewm_alpha_' + str(alpha).replace(".", "") + "_lag_" + str(lag)] = \
                dataframe["GRANDTOTAL_NEW"].transform(lambda x: x.shift(lag).ewm(alpha=alpha).mean())
    return dataframe