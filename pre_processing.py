import pandas as pd
import datetime as dt
import numpy as np
from currency_converter import CurrencyConverter
from run.pre_tools import rare_encoder
import seaborn as sns
import matplotlib.pyplot as plt



c = CurrencyConverter(fallback_on_missing_rate=True,fallback_on_wrong_date=True,decimal=True)

pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', 100)
pd.set_option('display.float_format', lambda x: '%.2f' % x)
pd.set_option('display.width', 500)

#Customer Order History DataSet
df_orders = pd.read_excel(r"C:\Users\kereviz\PycharmProjects\Customer Demand Forecasting\Valfsan2020-2022Siparişler.xlsx",sheet_name="Siparişler")

#Customer Foresights
df_foresights = pd.read_excel(r"C:\Users\kereviz\PycharmProjects\Customer Demand Forecasting\Valfsan2020-2022Siparişler.xlsx",sheet_name="Öngörüler")

#Material Types
df_mattype= pd.read_excel(r"C:\Users\kereviz\PycharmProjects\Customer Demand Forecasting\Valfsan2020-2022Siparişler.xlsx",sheet_name="Mattypes")

df_orders.isna().sum()

# Removing observations which SPRICE value is zero because they defactive
df_orders = df_orders[df_orders["SPRICE"] !=0 ]
df_orders = df_orders.reset_index()
df_orders.drop("index",axis=1)
# NaN values is not non values ,it is non-automative, fixed it with assign with 'Non-A'
df_orders.loc[df_orders["BRANCH"].isna(),"BRANCH"] = 'Non-A'
# We can keep dataset with few na observations, as we will use tree models.

#Items to drop
df_orders = df_orders.loc[~df_orders["MATERIAL"].isin(['0714960','9135003202'])]
#Fix item prices of material
df_orders["UNITPRICE"] = df_orders["SPRICE"] / df_orders["PRICEFACTOR"]
df_orders.loc[df_orders["UNITPRICE"] > 100,"SPRICE"] = df_orders.loc[df_orders["UNITPRICE"] > 100,"SPRICE"]/100
df_orders.loc[df_orders["MATERIAL"].isin(['TD106202',"TD106203"]),"PRICEFACTOR"] = 1000
df_orders.loc[(df_orders["MATERIAL"].isin(['LZ-02-DOR-01-AXX-0071','PK-02-DOR-01-AXX-0090']) & df_orders["PRICEFACTOR"]==1)]
df_orders.loc[((df_orders["DOCNUM"] == 20060269)) & (df_orders["ITEMNUM"].isin([60,110])),"PRICEFACTOR"] = 1000

df_orders = df_orders.reset_index()
df_orders.drop("index",axis=1)

# Calculate total sale price  of document item
df_orders["GRANDTOTAL"] = df_orders["QUANTITY"] * df_orders["SPRICE"]/ df_orders["PRICEFACTOR"]

# Calculating latency of order delivery
########################################################################################
########################################################################################
df_orders['DELIVERED_DATE'] = pd.to_datetime(df_orders['DELIVERED_DATE'], format='%Y-%m-%d')
df_orders['DUE_DATE'] = pd.to_datetime(df_orders['DUE_DATE'], format='%Y-%m-%d')
df_orders["LATENCY"] = [(df_orders["DELIVERED_DATE"][row] - df_orders['DUE_DATE'][row]).days for row in range(df_orders.shape[0])]

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
df_ordersnew = df_ordersnew.merge(df_mattype,how = 'left', on = 'MATERIAL')
df_ordersnew["GROUP"].value_counts()

# Encoding rare items before grouping by Month, CUSTOMER and GROUP
########################################################################################
########################################################################################
df_ordersnew = rare_encoder(df_ordersnew,"GROUP",0.005)
# Extracting monthly period of order time ( DUE_DATE)
df_ordersnew["Period"] = df_ordersnew["DUE_DATE"].dt.to_period('M')
# Converting Currency to 'EUR'
df_ordersnew["CURRENCY"] = ['TRY' if df_ordersnew["CURRENCY"][row] == 'TL' else df_ordersnew["CURRENCY"][row]
                            for row in range(df_ordersnew.shape[0])]
for row in range(len(df_ordersnew)):
    if df_ordersnew["GRANDTOTAL"][row] != 'EUR':
        # if df_ordersnew['DUE_DATE'][row] != pd.to_datetime('2018-01-13 00:00:00', format='%Y-%m-%d %H:%M:%S') :
        #     print(df_ordersnew['DUE_DATE'][row])
            df_ordersnew.loc[df_ordersnew.index == row,"GRANDTOTAL_NEW"] = c.convert(df_ordersnew["GRANDTOTAL"][row], df_ordersnew['CURRENCY'][row], 'EUR',
                                                date=df_ordersnew['DUE_DATE'][row])



df_ordersnew_backup = df_ordersnew.copy()
df_ordersnew = df_ordersnew_backup
df_ordersnew["GRANDTOTAL_NEW"] = df_ordersnew["GRANDTOTAL"]
df_ordersnew["GRANDTOTAL_NEW"].astype("float64")
df_ordersnew2 = df_ordersnew.groupby(["GROUP","GRCNAME1","Period","BRANCH",'COUNTRY']).GRANDTOTAL_NEW.sum()





df_ordersnew2 = df_ordersnew2.reset_index()

df_ordersnew2["Period"] = df_ordersnew2["Period"].astype(str)

#Visualizing

sns.lineplot(data=df_ordersnew2, x="Period", y="GRANDTOTAL_NEW", hue="GROUP",ci=None)
plt.xticks(rotation=90)
f = plt.figure()
f.set_figwidth(25)
f.set_figheight(20)
plt.show()

