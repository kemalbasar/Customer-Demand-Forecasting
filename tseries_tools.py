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


def smape(preds, target):
    n = len(preds)
    masked_arr = ~((preds == 0) & (target == 0))
    preds, target = preds[masked_arr], target[masked_arr]
    num = np.abs(preds - target)
    denom = np.abs(preds) + np.abs(target)
    smape_val = (200 * np.sum(num / denom)) / n
    return smape_val


def lgbm_smape(preds, train_data):
    labels = train_data.get_label()
    smape_val = smape(np.expm1(preds), np.expm1(labels))
    return 'SMAPE', smape_val, False

# Spliting data set into test and train sets.
# df_ordersnew2.to_csv("final_before_model.csv", index=False)


# test_stat, pvalue = shapiro(df.loc[df["group"] == "control", "Purchase"])
# print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))
# # p-value=0.5891
# # HO reddedilemez. Control grubunun değerleri normal dağılım varsayımını sağlamaktadır.
# #2.yol
# shapiro(df.loc[df.index[0:40],"Purchase"])
#
# test_stat, pvalue = shapiro(df.loc[df["group"] == "test", "Purchase"])
# print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))
# # p-value=0.1541
# # HO reddedilemez. Control grubunun değerleri normal dağılım varsayımını sağlamaktadır.
# #2.yol
# shapiro(df.loc[df.index[40::],"Purchase"])


