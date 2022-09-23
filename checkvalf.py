import pandas as pd
import datetime as dt
import numpy as np
from currency_converter import CurrencyConverter
import pre_tools as pt
import seaborn as sns
import matplotlib.pyplot as plt
import tseries_tools as tt

c = CurrencyConverter(fallback_on_missing_rate=True, fallback_on_wrong_date=True, decimal=True)

pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', 100)
pd.set_option('display.float_format', lambda x: '%.2f' % x)
pd.set_option('display.width', 500)

# Customer Order History DataSet
df_orders = pd.read_excel(
    r"C:\Users\kereviz\PycharmProjects\Customer Demand Forecasting\data_sources\Valfsan2020-2022Siparişler.xlsx",
    sheet_name="Siparişler")

# Customer Foresights
df_foresights = pd.read_excel(
    r"C:\Users\kereviz\PycharmProjects\Customer Demand Forecasting\data_sources\Valfsan2020-2022Siparişler.xlsx",
    sheet_name="Öngörüler")

# Material Types
df_mattype = pd.read_excel(
    r"C:\Users\kereviz\PycharmProjects\Customer Demand Forecasting\data_sources\Valfsan2020-2022Siparişler.xlsx",
    sheet_name="Mattypes")

df_orders.isna().sum()

# Removing observations which SPRICE value is zero because they defactive
df_orders = df_orders[df_orders["SPRICE"] != 0]
df_orders = df_orders.reset_index()
df_orders.drop("index", axis=1)
# NaN values is not non values ,it is non-automative, fixed it with assign with 'Non-A'
df_orders.loc[df_orders["BRANCH"].isna(), "BRANCH"] = 'Non-A'
# We can keep dataset with few na observations, as we will use tree models.

# Items to drop
df_orders = df_orders.loc[~df_orders["MATERIAL"].isin(['0714960', '9135003202'])]
# Fix item prices of material
df_orders["UNITPRICE"] = df_orders["SPRICE"] / df_orders["PRICEFACTOR"]
df_orders.loc[df_orders["UNITPRICE"] > 100, "SPRICE"] = df_orders.loc[df_orders["UNITPRICE"] > 100, "SPRICE"] / 100
df_orders.loc[df_orders["MATERIAL"].isin(['TD106202', "TD106203"]), "PRICEFACTOR"] = 1000
df_orders.loc[
    (df_orders["MATERIAL"].isin(['LZ-02-DOR-01-AXX-0071', 'PK-02-DOR-01-AXX-0090']) & df_orders["PRICEFACTOR"] == 1)]
df_orders.loc[(df_orders["DOCNUM"] == 20060269) & (df_orders["ITEMNUM"].isin([60, 110])), "PRICEFACTOR"] = 1000
df_orders.loc[((df_orders["DOCNUM"] == 21080256) & (df_orders["ITEMNUM"].isin([20, 40]))), "SPRICE"] = 348
df_orders = df_orders.reset_index()
df_orders.drop("index", axis=1)

# Calculate total sale price  of document item
df_orders["GRANDTOTAL"] = df_orders["QUANTITY"] * df_orders["SPRICE"] / df_orders["PRICEFACTOR"]

# Calculating latency of order delivery
########################################################################################
########################################################################################
df_orders['DELIVERED_DATE'] = pd.to_datetime(df_orders['DELIVERED_DATE'], format='%Y-%m-%d')
df_orders['DUE_DATE'] = pd.to_datetime(df_orders['DUE_DATE'], format='%Y-%m-%d')
df_orders["LATENCY"] = [(df_orders["DELIVERED_DATE"][row] - df_orders['DUE_DATE'][row]).days for row in
                        range(df_orders.shape[0])]

df_orders.describe().T

df_orders[df_orders["LATENCY"] > 100]

# 1.group by case : group wıth doc and ıtem num and calculate weıghted avarage of latencies

df_orders["PERC_OF_DELIVERED"] = df_orders["DELIVERED_QTY"] / df_orders["QUANTITY"]
df_orders["LATENCY_SHARE"] = df_orders["PERC_OF_DELIVERED"] * df_orders["LATENCY"]

df_ordersnew = df_orders.groupby(['DOCTYPE', 'DOCNUM', 'ITEMNUM',
                                  'COUNTRY', 'GRCNAME1', 'MATERIAL', 'MTEXT',
                                  'BRANCH', 'QUANTITY', 'SPRICE', 'GRANDTOTAL',
                                  'CURRENCY', 'DUE_DATE', 'DELIVERED_DATE', 'DELIVERED_QTY']).LATENCY_SHARE.sum()

df_ordersnew = df_ordersnew.reset_index()
########################################################################################
########################################################################################


# Merge material group table
df_ordersnew = df_ordersnew.merge(df_mattype, how='left', on='MATERIAL')
df_ordersnew["GROUP"].value_counts()

# Encoding rare items before grouping by Month, CUSTOMER and GROUP
########################################################################################
########################################################################################
df_ordersnew = pt.rare_encoder(df_ordersnew, "GROUP", 0.005)
# Extracting monthly period of order time ( DUE_DATE)
df_ordersnew["Period"] = df_ordersnew["DUE_DATE"].dt.to_period('M')
# Converting Currency to 'EUR'
df_ordersnew["CURRENCY"] = ['TRY' if df_ordersnew["CURRENCY"][row] == 'TL' else df_ordersnew["CURRENCY"][row]
                            for row in range(df_ordersnew.shape[0])]
for row in range(len(df_ordersnew)):
    if df_ordersnew["GRANDTOTAL"][row] != 'EUR':
        # if df_ordersnew['DUE_DATE'][row] != pd.to_datetime('2018-01-13 00:00:00', format='%Y-%m-%d %H:%M:%S') :
        #     print(df_ordersnew['DUE_DATE'][row])
        df_ordersnew.loc[df_ordersnew.index == row, "GRANDTOTAL_NEW"] = c.convert(df_ordersnew["GRANDTOTAL"][row],
                                                                                  df_ordersnew['CURRENCY'][row], 'EUR',
                                                                                  date=df_ordersnew['DUE_DATE'][row])

df_ordersnew_backup = df_ordersnew.copy()
df_ordersnew = df_ordersnew_backup
# We will work on Check Valves group at first.
df_checkvalf = df_ordersnew[df_ordersnew["GROUP"] == "CHKVLF"]
df_ordersnew = df_checkvalf

# df_ordersnew["GRANDTOTAL_NEW"] = df_ordersnew["GRANDTOTAL"]
df_ordersnew["GRANDTOTAL_NEW"] = df_ordersnew["GRANDTOTAL_NEW"].astype("float64")
# grouping by period
df_ordersnew2 = df_ordersnew.groupby(["GROUP", "Period","COUNTRY"]).GRANDTOTAL_NEW.sum()
# df_ordersnew["GRANDTOTAL_NEW"]

df_ordersnew2 = df_ordersnew2.reset_index()
df_ordersnew2["Period"] = df_ordersnew2["Period"].astype(str)
df_ordersnew2.sort_values(by='Period', inplace=True)
df_ordersnew2 = df_ordersnew2.reset_index()
df_ordersnew2 = df_ordersnew2.drop("index", axis=1)

########################################################################
########################################################################
# Visualizing
plt.figure(figsize=(13, 11), dpi=80)
plt.xticks(rotation=90)
sns.lineplot(data=df_ordersnew2, x="Period", y="GRANDTOTAL_NEW", hue="GROUP", ci=None)
plt.show()

for type in df_ordersnew2["GROUP"].unique():
    df_tolook = df_ordersnew2[df_ordersnew2["GROUP"] == type]
    plt.xticks(rotation=90)
    sns.lineplot(data=df_tolook, x="Period", y="GRANDTOTAL_NEW", ci=None)
    plt.title(type)
    plt.grid()
    plt.show()

########################################################################
########################################################################
# Visualizing2
df_orderspivot = df_ordersnew2.pivot_table("GRANDTOTAL_NEW", "Period", "GROUP")
df_orderspivot.columns

plt.figure(figsize=(25, 10))
plot_number = 1
for col in df_orderspivot.columns:
    if plot_number < 8:
        plt.subplot(1, 1, plot_number)
        sns.boxplot(x=df_orderspivot[col])
        plt.xlabel(col, fontsize=10)
        plt.title(f"Boxplot for {col}", fontsize=10)
    plot_number += 1
plt.tight_layout()
plt.show()

# Feature Extractşon
df_ordersnew2["Period"] = [dt.datetime.strptime(df_ordersnew2["Period"][row], '%Y-%m') for row in
                           range(df_ordersnew2.shape[0])]
df_ordersnew2['year'] = df_ordersnew2.Period.dt.year
df_ordersnew2['month'] = df_ordersnew2.Period.dt.month

# Creating Time Series Features
tt.lag_features(df_ordersnew2, [15, 10, 5, 3, 2])
tt.roll_mean_features(df_ordersnew2, [15, 10, 5, 3, 2])
tt.ewm_features(df_ordersnew2, [0.75, 0.65, 0.55, 0.45, 0.35], [15, 10, 5, 3, 2])

########################################################################
# One-Hot Encoding
df_ordersnew2.drop(["GROUP", "Period"], axis=1, inplace=True)

df_ordersnew2 = pd.get_dummies(df_ordersnew2, columns=["COUNTRY", "year", "month"])

df_ordersnew2["GRANDTOTAL_NEW"] = np.log1p(df_ordersnew2["GRANDTOTAL_NEW"].values)


# Spliting data set into test and train sets.
df_ordersnew2.to_csv("final_before_model.csv", index=False)




