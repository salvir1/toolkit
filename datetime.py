import pandas as pd
def convert_datetime(df):
    df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d', errors='ignore')
    return df