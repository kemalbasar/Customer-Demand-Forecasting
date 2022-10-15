import pandas as pd
import datetime as dt
import numpy as np
from currency_converter import CurrencyConverter
import seaborn as sns
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


from tools import tseries_tools as tt, pre_tools as pt

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

df_lmex = pd.read_excel(r"C:\Users\kereviz\PycharmProjects\Customer Demand Forecasting\data_sources\London-Metal-Exchange-Index-Historical-Data-Real.xlsx")

df_pmi = pd.read_excel(r"C:\Users\kereviz\PycharmProjects\Customer Demand Forecasting\data_sources\PMI-Index.xlsx")


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
df_orders.loc[(df_orders["MATERIAL"].isin(['LZ-02-DOR-01-AXX-0071', 'PK-02-DOR-01-AXX-0090']) & df_orders["PRICEFACTOR"] == 1)]
df_orders.loc[(df_orders["DOCNUM"] == 20060269) & (df_orders["ITEMNUM"].isin([60, 110])), "PRICEFACTOR"] = 1000
df_orders.loc[((df_orders["DOCNUM"] == 21080256) & (df_orders["ITEMNUM"].isin([20, 40]))), "SPRICE"] = 348
df_orders = df_orders.reset_index()
df_orders.drop("index", axis=1)
df_orders["GRANDTOTAL"] = df_orders["QUANTITY"] * df_orders["SPRICE"] / df_orders["PRICEFACTOR"]




# Calculate total sale price  of document item
df_orders["GRANDTOTAL"] = df_orders["QUANTITY"] * df_orders["SPRICE"] / df_orders["PRICEFACTOR"]

# Calculating latency of order delivery
########################################################################################
########################################################################################
#df_orders['DELIVERED_DATE'] = pd.to_datetime(df_orders['DELIVERED_DATE'], format='%Y-%m-%d')
df_orders['DUE_DATE'] = pd.to_datetime(df_orders['DUE_DATE'], format='%Y-%m-%d')

df_orders.describe().T

#df_orders[df_orders["LATENCY"] > 100]

# 1.group by case : group wıth doc and ıtem num and calculate weıghted avarage of latencies

#df_orders["PERC_OF_DELIVERED"] = df_orders["DELIVERED_QTY"] / df_orders["QUANTITY"]
#df_orders["LATENCY_SHARE"] = df_orders["PERC_OF_DELIVERED"] * df_orders["LATENCY"]

df_ordersnew = df_orders.groupby(['DOCTYPE', 'DOCNUM', 'ITEMNUM',
                                  'COUNTRY', 'GRCNAME1', 'MATERIAL', 'MTEXT',
                                  'BRANCH', 'QUANTITY', 'SPRICE', 'GRANDTOTAL',
                                  'CURRENCY', 'DUE_DATE','DELIVERED_QTY']).DELIVERED_DATE.count()

df_ordersnew = df_ordersnew.reset_index()
########################################################################################
########################################################################################

# Merge material group table
df_ordersnew = df_ordersnew.merge(df_mattype, how='left', on='MATERIAL')
df_ordersnew["GROUP"].value_counts()

# Getting only "check-valve" group
df_ordersnew_backup = df_ordersnew.copy()
df_ordersnew = df_ordersnew_backup
# We will work on Check Valves group at first.
df_checkvalf = df_ordersnew[df_ordersnew["GROUP"] == "CHKVLF"]
df_ordersnew = df_checkvalf
df_ordersnew = df_ordersnew.reset_index()
df_ordersnew.drop("index", axis=1, inplace=True)


# Rare Encoding
########################################################################################
########################################################################################
#df_ordersnew["GRCNAME1"] = ["EMERSON" if (("Emer" in val) | ("EMER" in val)) else val for val in
#                            df_ordersnew["GRCNAME1"]]
#df_ordersnew = pt.rare_encoder(df_ordersnew, "GRCNAME1", 0.05)
df_ordersnew = pt.rare_encoder(df_ordersnew, "COUNTRY", 0.05)
df_ordersnew.rename(columns={"GRCNAME1": "CUSTOMER"}, inplace=True)



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

# df_ordersnew["GRANDTOTAL_NEW"] = df_ordersnew["GRANDTOTAL"]
df_ordersnew["GRANDTOTAL_NEW"] = df_ordersnew["GRANDTOTAL_NEW"].astype("float64")


# grouping by period
df_ordersnew2 = df_ordersnew.groupby(["GROUP", "Period", "COUNTRY", ]).GRANDTOTAL_NEW.sum()
# df_ordersnew["GRANDTOTAL_NEW"]
df_ordersnew_back = df_ordersnew.copy()

df_ordersnew2 = df_ordersnew2.reset_index()
df_ordersnew2["Period"] = df_ordersnew2["Period"].astype(str)
df_ordersnew2.sort_values(by='Period', inplace=True)
df_ordersnew2 = df_ordersnew2.reset_index()
df_ordersnew2 = df_ordersnew2.drop("index", axis=1)

########################################################################
########################################################################
# Visualizing
def visualizing():
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

def feature_creating(df_ordersnew2):
    # Feature Extraction

    # Creating Time Series Features
    tt.lag_features(df_ordersnew2, [15, 12, 9, 8, 5, 3, 1])
    tt.roll_mean_features(df_ordersnew2, [15, 10, 8, 5, 2])
    tt.ewm_features(df_ordersnew2, [0.75, 0.65, 0.55, 0.45, 0.35, 0.25, 0.15, 0.05], [15, 10, 8, 5, 2])

    ########################################################################
    # One-Hot Encoding
    df_ordersnew2.drop(["GROUP", "Period"], axis=1, inplace=True)

    df_ordersnew2 = pd.get_dummies(df_ordersnew2, columns=["COUNTRY", "year", "month"])

    df_ordersnew2["GRANDTOTAL_NEW"] = np.log1p(df_ordersnew2["GRANDTOTAL_NEW"].values)
    return df_ordersnew2

visualizing()

df_ordersnew2["Period"] = [dt.datetime.strptime(df_ordersnew2["Period"][row], '%Y-%m') for row in
                           range(df_ordersnew2.shape[0])]
df_ordersnew2['year'] = df_ordersnew2.Period.dt.year
df_ordersnew2['month'] = df_ordersnew2.Period.dt.month
# Merging Metal Index Data
df_lmex['Date'] = pd.to_datetime(df_lmex['Date'], format='%m/%d/%Y')
df_lmex['year'] = df_lmex.Date.dt.year
df_lmex['month'] = df_lmex.Date.dt.month
df_lmex.drop('Date',axis=1,inplace=True)
df_ordersnew2 = df_ordersnew2.merge(df_lmex,left_on=["year","month"], right_on=["year","month"],how='left')

#Merging PMI INDEX based on Country
df_pmi.columns = ['Period', 'ACTUAL ', 'COUNTRY']
df_pmi['Period'] = pd.to_datetime(df_pmi['Period'], format='%m-%Y')
df_pmi['year'] = df_pmi.Period.dt.year
df_pmi['month'] = df_pmi.Period.dt.month
df_pmi.drop("Period",axis=1,inplace=True)
#renaming columns before merge ( BE(Belgium only country having export, and Thailand(THAI dominating rare encoded countries)
df_pmi["COUNTRY"] = df_pmi["COUNTRY"].apply(lambda x: "Rare" if x == 'THAI'  else ( 'BE' if x == 'EUR' else x))
df_ordersnew2 = df_ordersnew2.merge(df_pmi,how='left', left_on=["COUNTRY","year","month"], right_on = ["COUNTRY","year","month"])

df_ordersnew2 = feature_creating(df_ordersnew2)

#Savin final data set to excel for training
df_ordersnew2.to_excel(r"C:\Users\kereviz\PycharmProjects\Customer Demand Forecasting\data_sources\final_before_model.xlsx")

# pt.replace_missing_values(pt.df_ordersnew2, '0')

# for col in df_ordersnew2.columns:
#     if df_ordersnew2[col].dtype == 'O':
#         df_ordersnew2[col] = df_ordersnew2[col].astype("float64")
