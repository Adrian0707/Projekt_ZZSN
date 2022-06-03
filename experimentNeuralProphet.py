import os
import pickle
import argparse

from sklearn.model_selection import train_test_split
from utilsNN import *
from dataPreparation import data_prepare_day
from neuralprophet import NeuralProphet
from sklearn.preprocessing import MinMaxScaler

parser = argparse.ArgumentParser()

# RUN MODE
parser.add_argument('--mode', type=str, default='Predict')
parser.add_argument('--pred_periods', type=int, default=180)

# PREPARE DATASET
data_time_features = True
index = True
parser.add_argument('--prepare_data_before', type=bool, default=False)
parser.add_argument('--data_path', type=str, default=r'weatherHistory.csv')
parser.add_argument('--data_path_prep', type=str, default='../weatherHistory_prep_day_index.csv')

# LOAD SAVE PATHS
parser.add_argument('--save_model_dir', type=str, default='../SavedModels/neural_prophet_TB')
parser.add_argument('--load_model_path', type=str, default="..\SavedModels\\neural_prophet_TB\saved_model.pkl")
args = parser.parse_args()

if args.prepare_data_before:
    data_prepare_day(time_features=data_time_features, data_path=args.data_path, index=index)

os.makedirs(args.save_model_dir, exist_ok=True)

# DATA PREPARATION
df = pd.read_csv(args.data_path_prep)
df_date = pd.to_datetime(dict(year=df['Formatted Date'], month=df['Formatted Date.1'], day=df['Formatted Date.2']))
df = df.iloc[:, 20:27]

scaler = MinMaxScaler((-1, 1))
df[:] = scaler.fit_transform(df)

df['Date'] = df_date

train_data, valid_data = train_test_split(df[365:], shuffle=False, test_size=0.2)
print("train data shape is:", train_data.shape)
print("validation data shape is:", valid_data.shape)

variable_columns = df.columns.tolist()
variable_columns.remove('Date')

if args.mode == 'Train_load_model':
    with open(args.save_model_dir + '/saved_model.pkl', "rb") as f:
        m = pickle.load(f)
    print('model loaded')

if (args.mode == 'Train_new') | (args.mode == 'Train_load_model'):
    print('train')

    models = {}
    for col in variable_columns:
        models[col] = NeuralProphet()
        data = {'ds': train_data['Date'], 'y': train_data[col]}
        data = pd.DataFrame(data)
        models[col].fit(data)
        print('\n------------------------------------\n')
        print('Model for ' + col + ' done !')
        print('\n------------------------------------\n')
    with open(args.save_model_dir + '/saved_model.pkl', "wb") as f:
        pickle.dump(models, f)
else:
    print("predict")

    with open(args.load_model_path, "rb") as f:
        models = pickle.load(f)

    test_data = df[:365].reset_index()
    _, Testy, _ = sliding_windows_features(df[:365], args.pred_periods, args.pred_periods)

    mse_test_data = []
    predictions_y = []
    predictions_yhat1 = []

    for idx, var in enumerate(variable_columns):
        for i in range(len(Testy)):
            seq_true = Testy[i]
            y = seq_true[:, idx]

            data = pd.DataFrame({'ds': seq_true[:, 7]})
            data['y'] = None
            forecast = models[var].predict(data, decompose=False, raw=True)['step0'].tolist()

            summation = 0
            n = len(forecast)
            for i in range(0, n):
                difference = y[i] - forecast[i]
                squared_difference = difference ** 2
                summation = summation + squared_difference
            MSE = summation / n  #

            mse_test_data.append(MSE)

        predictions_y.append(y)
        predictions_yhat1.append(forecast)

    data_y = pd.DataFrame({variable_columns[0]: predictions_y[0], variable_columns[1]: predictions_y[1],
                           variable_columns[2]: predictions_y[2], variable_columns[3]: predictions_y[3],
                           variable_columns[4]: predictions_y[4], variable_columns[5]: predictions_y[5],
                           variable_columns[6]: predictions_y[6]})

    data_yhat = pd.DataFrame({variable_columns[0]: predictions_yhat1[0], variable_columns[1]: predictions_yhat1[1],
                              variable_columns[2]: predictions_yhat1[2], variable_columns[3]: predictions_yhat1[3],
                              variable_columns[4]: predictions_yhat1[4], variable_columns[5]: predictions_yhat1[5],
                              variable_columns[6]: predictions_yhat1[6]})

    y = scaler.inverse_transform(data_y)
    y_pred = scaler.inverse_transform(data_yhat)
    for idx, var in enumerate(variable_columns):
        plt.plot(y[:, idx], label='true')
        plt.plot(y_pred[:, idx], label='predict')
        plt.title(var)
        plt.legend()
        plt.show()

    test_mse = np.mean(mse_test_data)
    print("test mse continuous variables =", test_mse)
