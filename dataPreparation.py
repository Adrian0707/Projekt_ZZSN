import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


# Cyclical features function
def generate_cyclical_features(df, col_name, period, start_num=0):
    kwargs = {f'sin_{col_name}': lambda x: np.sin(2 * np.pi * (df[col_name] - start_num) / period),
              f'cos_{col_name}': lambda x: np.cos(2 * np.pi * (df[col_name] - start_num) / period)}
    return df.assign(**kwargs)


# One-hot Encoding function
def onehot_encode_pd(df, col_name):
    dummies = pd.get_dummies(df, columns=col_name, prefix=col_name)
    return dummies


# Data prepare function for hour step
def data_prepare_hour(data_dir):
    df = pd.read_csv(data_dir)

    # Set index to datetime and sort
    df = df.set_index(['Formatted Date'])
    df.index = pd.to_datetime(df.index, utc=True)
    if not df.index.is_monotonic:
        df = df.sort_index()

    # Drop Loud Cover one value "0"
    df = df.drop(columns=["Loud Cover"])

    df = df.drop(columns=["Daily Summary"])

    # One-hot Encoding for discrete attributes
    df['Precip Type'] = df['Precip Type'].fillna('NaN')

    df = onehot_encode_pd(df, ['Precip Type', 'Summary'])

    # Cyclical time features
    df = (df.assign(hour=df.index.hour).assign(day=df.index.day).assign(month=df.index.month).assign(
        week_of_year=df.index.isocalendar().week))

    df = generate_cyclical_features(df, 'hour', 24, 0)
    df = generate_cyclical_features(df, 'month', 12, 1)
    df = generate_cyclical_features(df, 'day', 31, 1)
    df = generate_cyclical_features(df, 'week_of_year', 53, 1)

    # Time features from timestamps
    df = onehot_encode_pd(df, ['hour', 'month', 'day', 'week_of_year'])

    pd.DataFrame(df).to_csv('weatherHistory_prep_hour.csv')

    return


# Data prepare function for day step
def data_prepare_day(time_features=True, data_path=r'weatherHistory.csv', index=False):
    df = pd.read_csv(data_path)

    # Set index to datetime and sort
    df = df.set_index(['Formatted Date'])
    df.index = pd.to_datetime(df.index, utc=True)
    if not df.index.is_monotonic:
        df = df.sort_index()

    df = df.drop(columns=["Loud Cover"])

    df = df.drop(columns=["Daily Summary"])
    df['Precip Type'] = df['Precip Type'].fillna('NaN')

    # Time features
    if time_features:
        df = (df.assign(day=df.index.day).assign(month=df.index.month).assign(week_of_year=df.index.isocalendar().week))
        df = generate_cyclical_features(df, 'month', 12, 1)
        df = generate_cyclical_features(df, 'day', 31, 1)
        df = generate_cyclical_features(df, 'week_of_year', 53, 1)
        df = onehot_encode_pd(df, ['month', 'day', 'week_of_year'])

    df1 = df.iloc[:, :2]
    df2 = df.iloc[:, 2:]
    mode = lambda x: x.mode() if len(x.mode()) < 2 else x.mode().iloc[0] if len(x) > 2 else np.array(x)
    df1 = df1.groupby(by=[df1.index.year, df1.index.month, df1.index.day]).agg(mode)
    df2 = df2.groupby(by=[df2.index.year, df2.index.month, df2.index.day]).mean()

    # One-hot Encoding for discrete attributes
    df1 = onehot_encode_pd(df1, ['Summary'])
    df1 = onehot_encode_pd(df1, ['Precip Type'])
    df = pd.merge(df1, df2, left_index=True, right_index=True)

    if index:
        pd.DataFrame(df).to_csv('weatherHistory_prep_day_index.csv', index=True)
    else:
        pd.DataFrame(df).to_csv('weatherHistory_prep_day.csv', index=False)
    return


# Data scale function
def scale_prepared_data(df):
    # MinMaxScaler for continuous attributes
    numeric_cols = ["Temperature (C)", "Apparent Temperature (C)", "Humidity", "Wind Speed (km/h)",
                    "Wind Bearing (degrees)", "Visibility (km)", "Pressure (millibars)"]
    scaler = MinMaxScaler((-1, 1))
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    return scaler

# def feature_label_split(df, target_col):
#     y = df[[target_col]]
#     X = df.drop(columns=[target_col])
#     return X, y
#
# # Split data function
# def train_val_test_split(df, target_col, test_ratio):
#     val_ratio = test_ratio / (1 - test_ratio)
#     X, y = feature_label_split(df, target_col)
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio, shuffle=False)
#     X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_ratio, shuffle=False)
#     return X_train, X_val, X_test, y_train, y_val, y_test
