########################
#Libraries
########################

import pandas as pd
import pathlib 
from io import StringIO



########################
#Data Read in function 
########################

def read_df(input):
    if isinstance(input, pd.DataFrame):
        return input
    
    try:
        path = pathlib.Path(input)

    except TypeError:
        raise TypeError("Input must be a DataFrame or a path to a CSV file")

    if path.is_file() and path.suffix == ".csv":
        return pd.read_csv(path)

    print("Error. Try pass a CSV direct or Filepath to a CSV.")
    return None



########################
#df checker 
#(should check for empty data frames and do some basic data Anonymisation)
########################
'''
def df_checker(data):
    #shape
    shape = data.shape
    #column names
    col_names = data.columns.tolist()

    #info as string
    buffer = StringIO()
    data.info(buf=buffer)
    info = buffer.getvalue()

    #missing values
    per_null= data.isna().mean() * 100
    num_null = data.isna().sum()



    #categorical columns
    cat_cols = data.select_dtypes(include='object').columns

    unique_val = {col: data[col].nunique() for col in cat_cols}
    num_cat_vals = {col: data[col].value_counts() for col in cat_cols}
    cat_col_proportion = {col: data[col].value_counts(normalize=True) for col in cat_cols}


    return {
            "shape": shape,
            "info": info,
            "percent_null": per_null,
            "num_null": num_null,
            "unique_values": unique_val,
            "cat_value_counts": num_cat_vals,
            "cat_value_proportion": cat_col_proportion,
            "col_names": col_names
        }
''' 




from io import StringIO
import pandas as pd
import numpy as np


def df_checker(data: pd.DataFrame) -> dict:
    # --- basic shape / columns ---
    shape = data.shape
    col_names = data.columns.tolist()

    # --- column type summary ---
    dtype_counts = data.dtypes.value_counts().to_dict()
    numeric_cols = data.select_dtypes(include="number").columns.tolist()
    cat_cols = data.select_dtypes(include="object").columns.tolist()

    # --- missing values (signals only) ---
    per_null = data.isna().mean() * 100
    cols_high_missing = per_null[per_null > 10].round(2).to_dict()
    cols_no_missing = per_null[per_null == 0].index.tolist()

    # --- categorical column signals ---
    cat_summary = {}

    for col in cat_cols:
        nunique = data[col].nunique(dropna=True)
        total = len(data[col])

        top_freq = data[col].value_counts(normalize=True, dropna=True).iloc[0] if total > 0 else 0

        cat_summary[col] = {
            "unique_values": int(nunique),
            "high_cardinality": bool(nunique > 50),
            "low_cardinality": bool(nunique <= 10),
            "top_value_proportion": round(float(top_freq), 3),
        }

    # --- numeric column signals ---
    num_summary = {}

    for col in numeric_cols:
        series = data[col].dropna()
        if series.empty:
            continue

        q1, q3 = series.quantile([0.25, 0.75])
        iqr = q3 - q1

        outlier_ratio = (
            ((series < q1 - 1.5 * iqr) | (series > q3 + 1.5 * iqr)).mean()
            if iqr > 0
            else 0
        )

        num_summary[col] = {
            "outlier_ratio": round(float(outlier_ratio), 3),
            "skewed": bool(abs(series.skew()) > 1),
        }

    return {
        "shape": shape,
        "column_names": col_names,
        "dtype_counts": dtype_counts,
        "missing_values": {
            "columns_high_missing_pct": cols_high_missing,
            "columns_no_missing": cols_no_missing,
        },
        "categorical_summary": cat_summary,
        "numeric_summary": num_summary,
    }

#anonymise data for security purposes
#DO NOT SEND SENSITIVE DATA INTO AN LLM EVER! I AM NOT LIABLE IF YOU DO THAT!
#WIP 

def data_anon(data):
    return data


########################
#Data Processing Prompt + Constraints
#(you can build registries in here that help guide the LLM)
########################




########################
#Prompt builder
#(this is where the registries and the prompt will be created)
########################


########################
#API Client call
########################