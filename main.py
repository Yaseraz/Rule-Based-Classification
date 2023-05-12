######################################
#   RULE BASED CLASSIFICATION
######################################

#############################
#        Libraries
#############################

import pandas as pd
from typing import Tuple

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)

#1 read persona.csv file and check general informations about data

persona = pd.read_csv("datasets/persona.csv")
df = persona.copy()
#############################
#          EDA
#############################

def check_df(dataframe: pd.DataFrame, shape: bool = True, columns: bool = True,
             info: bool = True, na: bool = True, desc: bool = True) -> Tuple:
    """
    This function checks DataFrame's Descriptions

    :param dataframe: A pandas dataframe
    :param shape: checks Dataframe's Shape
    :param columns: Prints Columns Names
    :param info: Prints Columns types
    :param na: Count and prints NaN values in DataFrame
    :param desc: Prints DataFrame's Descriptions
            like min, max, first, second, third quantile, Variance and Mean values
    :return: Data frame's Shape,
            Names of DataFrame's columns, DataFrame's columns Type,
            how many NA values in DataFrame, Descriptive Stats
    """
    outputs = []
    if shape:
        outputs.append(('Shape', dataframe.shape))
    if columns:
        outputs.append(('Columns', dataframe.columns))
    if info:
        outputs.append(('Types', dataframe.dtypes))
    if na:
        outputs.append(('NA', dataframe.isnull().sum()))
    if desc:
        outputs.append(('Descriptive', dataframe.describe().T))
    for output in outputs:
        print(15 * "#", output[0], 15 * "#")
        print(output[1], "\n")

    return tuple(outputs)


check_df(df)

### Check variables' classes and observation amounts
for i in df.columns:
    if i != ["AGE"]:
        df[i].value_counts()
    else:
        df[i].nunique()

### How much was earned in total from sales by country?
df.groupby("COUNTRY").agg({"PRICE": "sum"}).sort_values("PRICE",ascending=False)

### What are the sales numbers by SOURCE types?
df["SOURCE"].value_counts()

### What are the average prices by country?
df.groupby("COUNTRY").agg({"PRICE": "mean"}).sort_values("PRICE",ascending=False)

### What are the PRICE averages by SOURCE?
df.groupby("SOURCE").agg({"PRICE": "mean"})

### What are the PRICE averages in the COUNTRY-SOURCE breakdown?
df.groupby(["SOURCE", "COUNTRY"]).agg({"PRICE": "mean"})

#############################
#    DATA MANIPULATION
#############################

agg_df = df.groupby(["COUNTRY","SOURCE","SEX","AGE"]).agg({"PRICE":"mean"}).sort_values(by="PRICE",ascending=False)
agg_df.reset_index(inplace=True)
agg_df.head()


#Age has many different values but we want to create at most 5 different segmets
# for level based deffinitions. So we must create intervals for age variable.
bins=[0, 18, 24, 30, 40, 70]
labels = ["Child","Young","Middle_Age","Adult","Elder"]
agg_df["CAT_AGE"]= pd.cut(df["AGE"],right=True,bins= bins, labels= labels,
                      retbins=False,precision=0,include_lowest=True,
                      duplicates="drop", ordered=1)

#Now lets create level based custemers variable for our personas.

agg_df["customer_level_based"]= [row[0].upper() + "_" + row[1].upper() + "_" + row[2].upper() + "_" + row[5] for row in agg_df.values]

#Done but we have duplicated prices now.
agg_df["CUSTOMERS_LEVEL_BASED"].value_counts()

#Let's fix it

agg_df = agg_df.groupby("customer_level_based").\
                        agg({"PRICE" : "mean"}).\
                        sort_values("PRICE", ascending=False)

agg_df.head()
#A litle index problem occured.
agg_df.reset_index(inplace=True)
agg_df.head()

#We get rid of our all problems so lets get segments!

agg_df["SEGMENTS"] = pd.qcut(agg_df["PRICE"], q= 4 ,labels=["D","C","B","A"],
                             retbins=0,precision=2,duplicates="raise")
agg_df.head()

# Predicting Average Income
# define an american android user 27 y.o woman user and observe her avg income

new_user = [USA_ANDROID_FEMALE_24-30]
agg_df[agg_df["CUSTOMERS_LEVEL_BASED"] == new_user]

#############################
#       CONCLUSION
#############################


#As a result, we can classificate the new customer with rule-based classification,
#and we can observe what would be the segment of new customer
#and the possible average income what that user brings.

