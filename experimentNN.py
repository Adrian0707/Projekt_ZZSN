import os
import argparse
import torch.optim as optim

from sklearn.model_selection import train_test_split
from torch.autograd import Variable
from utilsNN import *
from dataPreparation import data_prepare_day, scale_prepared_data
from Models.LSTMSeq2SeqModelFeatures import Seq2SeqLSTM
from Models.GRUSeq2SeqModelFeatures import Seq2SeqGRU
from Models.RTransformerModel import RT

parser = argparse.ArgumentParser()

# RUN MODE
parser.add_argument('--mode', type=str, default='Train_new')  # 'Train_load_model', 'Train_new' or 'Predict'
parser.add_argument('--NN_type', type=str, default='GRU')  # 'LSTM' 'GRU' 'r_transformer'

# LOAD DATASET
data_time_features = True
parser.add_argument('--prepare_data_before', type=bool, default=False)
parser.add_argument('--data_time_features', type=bool, default=True)
parser.add_argument('--data_path', type=str, default=r'weatherHistory.csv')
parser.add_argument('--data_path_prep', type=str, default=r'weatherHistory_prep_day.csv')

# PARAMETERS
n_features_out = 24
parser.add_argument('--in_seq_length', type=int, default=180)
parser.add_argument('--out_seq_length', type=int, default=180)
parser.add_argument('--epochs', type=int, default=100)

# LSTM & GRU PARAMETERS
parser.add_argument('--lr_GRU_LSTM', type=float, default=4e-4)
parser.add_argument('--weight_decay', type=float, default=1e-6)
parser.add_argument('--embedding_dim_GRU_LSTM', type=int, default=16)
parser.add_argument('--encoder_num_layers_GRU_LSTM', type=int, default=3)
parser.add_argument('--encoder_dropout_GRU_LSTM', type=float, default=0.35)
parser.add_argument('--decoder_num_layers_GRU_LSTM', type=int, default=3)
parser.add_argument('--decoder_dropout_GRU_LSTM', type=float, default=0.35)

# R-TRANSFORMER PARAMETERS
parser.add_argument('--dropout', type=float, default=0.1)
parser.add_argument('--clip', type=float, default=0.15)
parser.add_argument('--ksize', type=int, default=6)
parser.add_argument('--n_level', type=int, default=3)
parser.add_argument('--log-interval', type=int, default=100, metavar='N')
parser.add_argument('--lr_r_transformer', type=float, default=5e-05)
parser.add_argument('--optim', type=str, default='SGD')
parser.add_argument('--rnn_type', type=str, default='GRU')
parser.add_argument('--d_model', type=int, default=160)
parser.add_argument('--n', type=int, default=1)
parser.add_argument('--h', type=int, default=4)

# LOAD MODEL
parser.add_argument('--load_model_path', type=str,
                    default='SavedModels\GRU_in180_out180_data_features_newArch_TB\in180_out180_time_features_True_lr=0.0004_weight_decay=1e-06_epochs=100_embed_dim=16_dropout=0.35_num_layers=3_losse_val=2.389555910205584.pt')
args = parser.parse_args()

if args.prepare_data_before:
    data_prepare_day(time_features=data_time_features, data_path=args.data_path)

in_seq_length = args.in_seq_length
out_seq_length = args.out_seq_length

if args.NN_type == 'r_transformer':
    out_seq_length = in_seq_length

# LOAD SAVE PATHS
save_model_dir = 'SavedModels/' + args.NN_type + '_in' + str(in_seq_length) + '_out' + str(
    out_seq_length) + '_data_features_newArch_TB_test'

os.makedirs(save_model_dir, exist_ok=True)

save_model_name = "in" + str(in_seq_length) + "_out" + str(out_seq_length) + "_time_features_" + str(
    data_time_features) + "_lr=" + str(args.lr_GRU_LSTM) + "_weight_decay=" + str(args.weight_decay) + "_epochs=" + str(
    args.epochs) + "_embed_dim=" + str(args.embedding_dim_GRU_LSTM) + "_dropout=" + str(
    args.decoder_dropout_GRU_LSTM) + "_num_layers=" + str(args.encoder_num_layers_GRU_LSTM)

save_model_path = save_model_dir + "\\" + save_model_name

# DATA PREPARATION
df = pd.read_csv(args.data_path_prep)
to_scale = df.iloc[:, 17:24]
scaler = scale_prepared_data(to_scale)
df.iloc[:, 17:24] = to_scale

print("n_features = ", len(df.columns))
n_features_in = len(df.columns)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("device is:", device)

train_data, valid_data = train_test_split(df[365:], shuffle=False, test_size=0.2)
print("train data shape is:", train_data.shape)
print("validation data shape is:", valid_data.shape)

train_X, train_y, train_features = sliding_windows_features(train_data, in_seq_length, out_seq_length)
valid_X, valid_y, valid_features = sliding_windows_features(valid_data, in_seq_length, out_seq_length)

if data_time_features:
    train_y = np.array(train_y)[:, :, :n_features_out]
    valid_y = np.array(valid_y)[:, :, :n_features_out]
else:
    train_y = np.array(train_y)
    valid_y = np.array(valid_y)

if args.NN_type == 'r_transformer':
    train_X_r_transformer = weather_to_r_transformer_shape(np.array(train_X))
    valid_X_r_transformer = weather_to_r_transformer_shape(np.array(valid_X))

    train_y_r_transformer = weather_to_r_transformer_shape(train_y)
    valid_y_r_transformer = weather_to_r_transformer_shape(valid_y)

if args.NN_type == 'LSTM':
    model = Seq2SeqLSTM(in_seq_length, n_features_in, device, output_length=out_seq_length,
                        embedding_dim=args.embedding_dim_GRU_LSTM, encoder_num_layers=args.encoder_num_layers_GRU_LSTM,
                        encoder_dropout=args.encoder_dropout_GRU_LSTM,
                        decoder_num_layers=args.decoder_num_layers_GRU_LSTM,
                        decoder_dropout=args.decoder_dropout_GRU_LSTM)
elif args.NN_type == 'GRU':
    model = Seq2SeqGRU(in_seq_length, n_features_in, device, output_length=out_seq_length,
                       embedding_dim=args.embedding_dim_GRU_LSTM, encoder_num_layers=args.encoder_num_layers_GRU_LSTM,
                       encoder_dropout=args.encoder_dropout_GRU_LSTM,
                       decoder_num_layers=args.decoder_num_layers_GRU_LSTM,
                       decoder_dropout=args.decoder_dropout_GRU_LSTM)
elif args.NN_type == 'r_transformer':
    dropout = args.dropout
    emb_dropout = args.dropout
    model = RT(n_features_in, args.d_model, n_features_out, h=args.h, rnn_type=args.rnn_type, ksize=args.ksize,
               n_level=args.n_level, n=args.n, dropout=dropout, emb_dropout=emb_dropout)

model = model.to(device)
print(model)

if args.mode == 'Train_load_model':
    model.load_state_dict(torch.load(args.load_model_path, map_location=device))
    model.eval()
    print('model loaded')
else:
    if (args.NN_type == 'LSTM') | (args.NN_type == 'GRU'):
        model.apply(init_weights)

if (args.mode == 'Train_new') | (args.mode == 'Train_load_model'):
    criterion_continuous = torch.nn.MSELoss().to(device)
    criterion_binary = torch.nn.CrossEntropyLoss().to(device)

    if (args.NN_type == 'LSTM') | (args.NN_type == 'GRU'):
        trainX = Variable(torch.Tensor(np.array(train_X)))
        trainy = Variable(torch.Tensor(np.array(train_y)))
        train_features = Variable(torch.Tensor(np.array(train_features)))

        validX = Variable(torch.Tensor(np.array(valid_X)))
        validy = Variable(torch.Tensor(np.array(valid_y)))
        valid_features = Variable(torch.Tensor(np.array(valid_features)))

        print("trainX shape is:", trainX.size())
        print("trainy shape is:", trainy.size())
        print("train features  shape is:", train_features.size())

        print("validX shape is:", validX.size())
        print("validy shape is:", validy.size())
        print("valid features  shape is:", valid_features.size())

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr_GRU_LSTM, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5e-3, eta_min=1e-8, last_epoch=-1)

        model, history = train_model_features_GRU_LSTM(model, trainX, trainy, train_features, validX, validy,
                                                       valid_features, in_seq_length, n_epochs=args.epochs,
                                                       device=device, optimizer=optimizer,
                                                       criterion_binary=criterion_binary,
                                                       criterion_continuous=criterion_continuous, scheduler=scheduler,
                                                       save_model_path=save_model_path)
    elif args.NN_type == 'r_transformer':
        lr_r_transformer = args.lr_r_transformer
        optimizer = getattr(optim, args.optim)(model.parameters(), lr=lr_r_transformer)
        model, history = train_model_features_r_transformer(model, train_X_r_transformer, train_y_r_transformer,
                                                            valid_X_r_transformer, valid_y_r_transformer, args.epochs,
                                                            device, criterion_continuous, criterion_binary, optimizer,
                                                            save_model_path, args.clip, lr_r_transformer)

else:
    print("predict")

    TestX, Testy, Test_features = sliding_windows_features(df[:365], in_seq_length, out_seq_length)
    if (args.NN_type == 'GRU') | (args.NN_type == 'LSTM'):
        TestX = Variable(torch.Tensor(np.array(TestX)))
        Testy = Variable(torch.Tensor(np.array(Testy)))
        Test_features = Variable(torch.Tensor(np.array(Test_features)))
    elif args.NN_type == 'r_transformer':
        TestX = weather_to_r_transformer_shape(np.array(TestX))
        Testy = weather_to_r_transformer_shape(np.array(Testy))

    model.load_state_dict(torch.load(args.load_model_path, map_location=device))
    model.eval()
    criterion_continuous = torch.nn.MSELoss().to(device)
    criterion_binary = torch.nn.CrossEntropyLoss().to(device)
    if (args.NN_type == 'GRU') | (args.NN_type == 'LSTM'):
        predict_example_GRU_LSTM(model, device, scaler, criterion_continuous, criterion_binary, df, TestX, Testy,
                                 Test_features, in_seq_length, out_seq_length)
    elif args.NN_type == 'r_transformer':
        predict_example_r_transformer(model, device, scaler, criterion_continuous, criterion_binary, df, TestX, Testy,
                                      out_seq_length)
