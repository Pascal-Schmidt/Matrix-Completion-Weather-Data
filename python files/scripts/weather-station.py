import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import missing_gap as mg

# import testing and training data
train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")


cols = ["WINDDIR", "WINDSPEED", "TEMPERATURE", "DEWPOINT", "PRESSURE"]
np.warnings.filterwarnings('ignore')

# sometimes, values are above 4 and below -4. Because the data was standardized
# beforehand, we impute NA for values above 4 and below 4.
for i in range(len(cols)):
    a = np.array([train[cols[i]]])
    a = np.where( (a > 4) | (a < -4), np.nan, a)
    a = a.tolist()
    train[cols[i]] = a[0]

# split ID column into rows and columns
new = test["ID"].str.split("-", n = 1, expand = True)
test["row"] = new[0]
test["col"] = new[1]
test = test.drop(columns = ["ID"])

# assign column names
test.loc[test["col"] == "1", "col"] = "WINDDIR"
test.loc[test["col"] == "2", "col"] = "WINDSPEED"
test.loc[test["col"] == "3", "col"] = "TEMPERATURE"
test.loc[test["col"] == "4", "col"] = "DEWPOINT"
test.loc[test["col"] == "5", "col"] = "PRESSURE"
test["value"] = np.nan

df = test
df["above"] = np.nan
df["below"] = np.nan

# helper function which determines how many missing values are above and below
# a given value, appearing in the test data set, in the training data set
df = mg.missing_gap(df)

# filter out values that have more than six missing values above and below
df = df[(df['above'] > 6) & (df['below'] > 6)]
df.index = range(len(df))
df["row"] = pd.to_numeric(df["row"])

# use interpolation for values that have 6 or less missing values preceding and following them
train[cols] = train.interpolate(method = "linear",
                                limit = 6,
                                limit_direction = "both")[cols]

# fill out test data set with imputed values from training data set
for i in range(len(test)):
    test.loc[i, "value"] = train.loc[int(test["row"][i]) - 1,
                                     test["col"][i]]

reg_df = train

# coerce MONTH to categorical variable
reg_df["MONTH"] = reg_df["MONTH"].astype('category')

# one hot encoding for MONTH column for predictor
dummies = pd.get_dummies(reg_df.MONTH)
reg_df = reg_df.join(dummies)
reg_df = reg_df.drop("MONTH", 1)

for i in range(len(df)):

    # identify all values in row where we have to predict, which
    # predictors are not missing
    x = train.loc[df["row"][i], cols].isnull()
    x = x.to_frame().T

    # extract predictors from data frame x, which are not missing
    j = 0
    names = {}
    for k in range(len(x.columns)):
        if (x.iloc[0][x.columns[k]]) == False:
            names[j] = x.columns[k]
            j = j + 1

    # if all predictors in row are missing use mean imputation by station
    station = train.loc[df["row"][i], :]["USAF"]
    month = train.loc[df["row"][i], :]["MONTH"]
    if (len(names) == 0):

        d = train[(train["USAF"] == station) &
                  (train["MONTH"] == month)]

        df.loc[i, "value"] = np.nanmean(d.loc[:, df["col"][i]])

    else:
        next

    # check if predictors have more than 40% missing values in column
    # if they do, discard that particular predictor
    temp = train[(train["USAF"] == station)][names.values()]

    prop = {}
    j = 0
    for k in range(len(temp.columns)):
        prop[j] = temp[temp.columns[k]].isnull().sum() / len(temp)
        j = j + 1

    preds = np.array(list(prop.values())) < 0.4
    names = temp[temp.columns[preds]]

    # if there are no predictor_vars because there are more then 40%
    # of missing values in each predictor column, then just impute the
    # average of the response column by station

    if (len(temp.columns) == 0):

        station = train.loc[df["row"][i], :]["USAF"]
        month = train.loc[df["row"][i], :]["MONTH"]

        d = train[(train["USAF"] == station) &
                  (train["MONTH"] == month)]

        df.loc[i, "value"] = np.nanmean(d.loc[:, df["col"][i]])

    else:
        next

    # add predictor MONTH and response variable to list
    names = list(names.columns)
    names.extend(["MONTH", "USAF", df["col"][i]])

    names.remove("MONTH")
    names.extend([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])

    # if there are enough observations (500 or more) for a particular station,
    # we do a linear regression by station
    lm_df = reg_df[names][reg_df["USAF"] == station].dropna()

    if (len(lm_df) > 500):

        # response variable
        Y = lm_df[df["col"][i]]
        names.remove(df["col"][i])
        names.remove("USAF")

        # predictor data frame
        X = lm_df[names]

        new_data = reg_df.loc[df["row"][i]].to_frame().T
        new_data = new_data[names]

        # linear regression model
        lm = LinearRegression()
        lm.fit(X, Y)

        # predictions
        try:
            pred_value = list(lm.predict(new_data))
            df.loc[i, "value"] = pred_value

        except:
            print("prediction", i, "failed")

    else:

        # if there are NOT enough observations (500 or more) for a particular station,
        # we do a linear regression with station as categorical variable
        lm_df = reg_df[names].dropna()

        Y = lm_df[df["col"][i]]
        names.remove(df["col"][i])
        reg_df_2 = reg_df

        # one hot encoding for USAF column for predictor
        dummies = pd.get_dummies(reg_df_2.USAF)
        reg_df_2 = reg_df_2.join(dummies)
        reg_df_2 = reg_df_2.drop("USAF", 1)

        # predictor data frame
        X = lm_df[names]

        new_data = reg_df.loc[df["row"][i]].to_frame().T
        new_data = new_data[names]

        # predictions
        try:
            pred_value = list(lm.predict(new_data))
            df.loc[i, "value"] = pred_value

        except:
            print("prediction", i, "failed")

# fill out test data set NA values with imputed values from df data set
test["row"] = pd.to_numeric(test["row"])
for i in range(len(df)):
    test.loc[test["row"] == df["row"][i], "value"] = df["value"][i]

# finally, fill out test data set NA values with imputed values in training data set
for i in range(len(test)):
    if(pd.isnull(test.loc[i, "value"])):
        test.loc[i, "value"] = train.loc[test["row"][i] - 1,
                                         test["col"][i]]

# for values, where linear regression failed or for which we have not done imputation yet
# use mean imputation across all stations for same month
df = test[pd.isnull(test["value"])]
df.index = range(len(df))

for i in range(len(df)):
    month = train.loc[df["row"][i], :]["MONTH"]
    d = train[(train["MONTH"] == month)]
    df.loc[i, "value"] = np.nanmean(d.loc[:, df["col"][i]])

for i in range(len(df)):
    test.loc[test["row"] == df["row"][i], "value"] = df["value"][i]

test.loc[test["col"] == "WINDDIR", "col"] = "1"
test.loc[test["col"] == "WINDSPEED", "col"] = "2"
test.loc[test["col"] == "TEMPERATURE", "col"] = "3"
test.loc[test["col"] == "DEWPOINT", "col"] = "4"
test.loc[test["col"] == "PRESSURE", "col"] = "5"
test["ID"] = test["row"].map(str) + "-" + test["col"]
test = test[["ID", "value"]]

# if values in vale column are too large or too small impute 0
col = np.array(test["value"])
col = np.where( (col > 5) | (col < -5), 0, col)
col = col.tolist()
test["value"] = col

 # write csv
test.to_csv("results.csv", index = False)