import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler


def grab_col_name(df):
    cat_vars = [col for col in df.columns if str(df[col].dtypes) in ["category", "object", "bool"]]
    num_but_cat = [col for col in df.columns if
                   (str(df[col].dtypes) in ["int64", "int32", "float32", "float64"]) & (df[col].nunique() < 10)]
    cat_vars = cat_vars + num_but_cat
    num_vars = [col for col in df.columns if col not in cat_vars]
    cat_but_car = [col for col in df.columns if
                   (str(df[col].dtypes) in ["category", "object", "bool"]) & (df[col].nunique() > 30)]

    len_of_carvars = len(cat_but_car)
    len_of_numvars = len(num_vars)
    len_of_catvars = len(cat_vars)

    print('Categoric Vars = %d ' % len_of_catvars)
    print('Numeric Vars = %d ' % len_of_numvars)
    print('Cardinal Vars = %d ' % len_of_carvars)

    print('Categoric Vars: ' + str(cat_vars))
    print('Numeric Vars: ' + str(num_vars))
    print('Cardinal Vars: ' + str(cat_but_car))

    return cat_vars, num_vars, cat_but_car


def cat_summary(df, column, plot=False, targetvar=False):
    """
    It describe the spesifications of variable.

    Parameters
    ----------
    df: Dataframe
        Dataframe you want to analyse
    column: string
        the column which you will get summary of
    plot: bool
        parameter which decide plot gonne show or not
    targetvar: string
        name of target variable analyse


    Returns
    -------

    """
    print(pd.DataFrame({column: df[column].value_counts(),
                        'rate': df[column].value_counts() / len(df)}))
    print("###############################################")

    if targetvar:
        print(df.pivot_table(targetvar, column))

    if plot:
        sns.countplot(x=df[column], data=df)
        plt.show(block=True)


def num_summary(dataframe, column, plot=False, plot_type="box", targetvar=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[column].describe(quantiles).T)

    if targetvar:
        print(dataframe.pivot_table(targetvar, column))

    if plot:
        if plot_type == "hist":
            dataframe[column].hist()
        elif plot_type == "box":
            dataframe[[column]].boxplot()

        plt.xlabel(column)
        plt.title(column)
        plt.show(block=True)


def target_summary_with_cat(dataframe, target, categorical_col):
    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean()}), end="\n\n\n")


def get_core_triangle(dataframe):
    core = dataframe.corr()
    core = core.abs()
    core_triangle = core.where(np.triu(np.ones(core.shape), k=1).astype(np.bool_))
    return core_triangle


def high_correlated_cols(dataframe, plot=False, corr_th=0.70):
    upper_triangle_matrix = get_core_triangle(dataframe)
    drop_list = [upper_triangle_matrix.loc[(upper_triangle_matrix[col] > corr_th), col] for col in
                 upper_triangle_matrix.columns if any(upper_triangle_matrix[col] > corr_th)]

    if plot:
        sns.set(rc={'figure.figsize': (15, 15)})
        sns.heatmap(dataframe.corr(), cmap="RdBu")
        plt.show()
    return drop_list


def outlier_thresholds(variable, q1=0.05, q3=0.95):
    if type(variable) == pd.core.series.Series:
        quartile1 = variable.quantile(q1)
        quartile3 = variable.quantile(q3)
        interquantile_range = quartile3 - quartile1
        up_limit = quartile3 + 1.5 * interquantile_range
        low_limit = quartile1 - 1.5 * interquantile_range
        return low_limit, up_limit
    else:
        raise Exception("I need pandas series object!")


def replace_with_thresholds(dataframe, variable, q1=0.05, q3=0.95):
    low_limit, up_limit = outlier_thresholds(dataframe[variable], q1, q3)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = round(low_limit, 0)
    dataframe.loc[(dataframe[variable] > up_limit), variable] = round(up_limit, 0)


def check_outlier(dataframe, col_name, q1=0.05, q3=0.95):
    low_limit, up_limit = outlier_thresholds(dataframe[col_name], q1, q3)
    if dataframe[(dataframe[col_name] > round(up_limit, 0)) | (dataframe[col_name] < round(low_limit, 0))].any(
            axis=None):
        return True
    else:
        return False


def remove_outlier(dataframe, col_name, q1=0.01, q3=0.99):
    low_limit, up_limit = outlier_thresholds(dataframe[col_name], q1, q3)
    df_without_outliers = dataframe[~((dataframe[col_name] < low_limit) | (dataframe[col_name] > up_limit))]
    return df_without_outliers


def grab_outliers(dataframe, col_name, index=False):
    low, up = outlier_thresholds(dataframe[col_name])

    if dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].shape[0] > 10:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].head())
    else:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))])

    if index:
        outlier_index = dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].index
        return outlier_index


def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    # Deişken tipini de ekleyelim buraya.
    listof_variale_types = grab_col_name(dataframe)
    keys = ["cat", "num", "car"]
    df_types = pd.DataFrame.from_dict({keys[i]: listof_variale_types[i] for i in range(2)}, orient="index").T

    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2), 0], axis=1, keys=['n_miss', 'ratio'])
    missing_df['type'] = 0

    for i in range(missing_df.shape[0]):

        if df_types.cat.str.contains(missing_df.iloc[0].name).sum() > 0:
            missing_df.iloc[[i], [2]] = 'cat'
        elif df_types.cat.str.contains(missing_df.iloc[0].name).sum() > 0:
            missing_df.iloc[[i], [2]] = 'num'
        else:
            missing_df.iloc[[i], [2]] = 'car'

    print(missing_df, end="\n")
    return missing_df


def quick_missing_imp(data, num_method="median", cat_length=20, target="SalePrice"):
    variables_with_na = [col for col in data.columns if
                         data[col].isnull().sum() > 0]  # Eksik değere sahip olan değişkenler listelenir

    temp_target = data[target]

    print("# BEFORE")
    print(data[variables_with_na].isnull().sum(), "\n\n")  # Uygulama öncesi değişkenlerin eksik değerlerinin sayısı

    # değişken object ve sınıf sayısı cat_lengthe eşit veya altındaysa boş değerleri mode ile doldur
    data = data.apply(lambda x: x.fillna(x.mode()[0]) if (x.dtype == "O" and len(x.unique()) <= cat_length) else x,
                      axis=0)

    # num_method mean ise tipi object olmayan değişkenlerin boş değerleri ortalama ile dolduruluyor
    if num_method == "mean":
        data = data.apply(lambda x: x.fillna(x.mean()) if x.dtype != "O" else x, axis=0)
    # num_method median ise tipi object olmayan değişkenlerin boş değerleri ortalama ile dolduruluyor
    elif num_method == "median":
        data = data.apply(lambda x: x.fillna(x.median()) if x.dtype != "O" else x, axis=0)

    data[target] = temp_target

    print("# AFTER \n Imputation method is 'MODE' for categorical variables!")
    print(" Imputation method is '" + num_method.upper() + "' for numeric variables! \n")
    print(data[variables_with_na].isnull().sum(), "\n\n")

    return data


# agg_df['customers_level_based'] = (agg_df[['Nationality', 'Channel', "age_cat"]].agg('_'.join, axis=1)).str.upper()
# df["cat_totalbill"] = pd.cut(df["total_bill"], [0, 10, 20, 30, 40, 50, df["total_bill"].max()] , labels = ["0_10",
# "10_20","20_30","30_40","40_50","50_60"])

def rare_analyser(dataframe, target, cat_cols):
    for col in cat_cols:
        print(col, ":", len(dataframe[col].value_counts()))
        print(pd.DataFrame({"COUNT": dataframe[col].value_counts(),
                            "RATIO": dataframe[col].value_counts() / len(dataframe),
                            "TARGET_MEAN": dataframe.groupby(col)[target].mean()}), end="\n\n\n")


def rare_encoder(dataframe, cat, rare_perc):
    temp_df = dataframe.copy()

    rare_columns = [col for col in temp_df.columns if col in cat
                    and (temp_df[col].value_counts() / len(temp_df) < rare_perc).any(axis=None)]
    print(rare_columns)
    for var in rare_columns:
        tmp = temp_df[var].value_counts() / len(temp_df)
        rare_labels = tmp[tmp < rare_perc].index
        temp_df[var] = np.where(temp_df[var].isin(rare_labels), 'Rare', temp_df[var])

    return temp_df


def binary_encoder(df):
    binary_col = [col for col in df.columns if df[col].dtype not in [int, float]
                  and df[col].nunique() == 2]

    for col in binary_col:
        labelencoder = LabelEncoder()
        df[col] = labelencoder.fit_transform(df[col])
    # labelencoder.inverse_transform([])
    return df

def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe


# def local_outlier_plot(df,X_scores):
#     plt.scatter(df[:, 0], df[:, 1], color="k", s=3.0, label="Data points")
#     # plot circles with radius proportional to the outlier scores
#     radius = (X_scores.max() - X_scores) / (X_scores.max() - X_scores.min())
#     plt.scatter(
#         df[:, 0],
#         df[:, 1],
#         s=1000 * radius,
#         edgecolors="r",
#         facecolors="none",
#         label="Outlier scores",
#     )
#     plt.axis("tight")
#     plt.xlim((-5, 5))
#     plt.ylim((-5, 5))
#     plt.xlabel("prediction errors: %d" % (n_errors))
#     legend = plt.legend(loc="upper left")
#     legend.legendHandles[0]._sizes = [10]
#     legend.legendHandles[1]._sizes = [20]
#     plt.show()
