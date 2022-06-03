import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd
from torchmetrics.classification import Accuracy
from torchmetrics.regression import MeanAbsoluteError
from torch.utils.tensorboard import SummaryWriter
from fastprogress import master_bar, progress_bar


def plot_loss_update(epoch, train_loss, valid_loss, acc_val_summary, acc_val_precip_type, acc_train_summary,
                     acc_train_precip_type, mse_train_continuous, mse_val_continuous, model_path):
    x = range(1, epoch + 1)

    x_margin = 0.1
    y_margin = 0.05
    plt.margins(x=x_margin, y=y_margin)
    plt.plot(x, train_loss, color='r', label='train loss')
    plt.plot(x, valid_loss, color='g', label='valid loss')
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.savefig(model_path + "loss.jpg")
    plt.close()

    x_margin = 0.1
    y_margin = 0.05
    plt.margins(x=x_margin, y=y_margin)
    plt.plot(x, acc_train_summary, color='r', label='train accuracy summary')
    plt.plot(x, acc_val_summary, color='g', label='valid accuracy summary')
    plt.xlabel("epoch")
    plt.ylabel("accuracy summary")
    plt.legend()
    plt.savefig(model_path + "accuracy_summary.jpg")
    plt.close()

    x_margin = 0.1
    y_margin = 0.05
    plt.margins(x=x_margin, y=y_margin)
    plt.plot(x, acc_train_precip_type, color='r', label='train accuracy precip type')
    plt.plot(x, acc_val_precip_type, color='g', label='valid accuracy precip type')
    plt.xlabel("epoch")
    plt.ylabel("accuracy precip type")
    plt.legend()
    plt.savefig(model_path + "accuracy_precip_type.jpg")
    plt.close()

    x_margin = 0.1
    y_margin = 0.05
    plt.margins(x=x_margin, y=y_margin)
    plt.plot(x, mse_train_continuous, color='r', label='train mse continuous')
    plt.plot(x, mse_val_continuous, color='g', label='valid mse continuous')
    plt.xlabel("epoch")
    plt.ylabel("mse continuous")
    plt.legend()
    plt.savefig(model_path + "mse_continuous.jpg")
    plt.close()


def train_model(model, TrainX, Trainy, ValidX, Validy, seq_length, n_epochs, device, criterion_continuous,
                criterion_binary, optimizer, scheduler, save_model_path):
    history = dict(train=[], val=[])

    # best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 10000.0
    mb = master_bar(range(1, n_epochs + 1))

    for epoch in mb:
        model = model.train()

        train_losses = []
        for i in progress_bar(range(TrainX.size()[0]), parent=mb, ):
            seq_inp = TrainX[i, :, :].to(device)
            seq_true = Trainy[i, :, :].to(device)

            optimizer.zero_grad()
            seq_pred = model(seq_inp, seq_inp[seq_length - 1:seq_length, :])

            loss = 10 * criterion_continuous(seq_pred[:, 17:24], seq_true[:, 17:24]) + criterion_binary(
                seq_pred[:, :14], seq_true[:, :14]) + criterion_binary(seq_pred[:, 14:17], seq_true[:, 14:17])

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()

            train_losses.append(loss.item())

        val_losses = []
        model = model.eval()
        with torch.no_grad():
            for i in progress_bar(range(ValidX.size()[0]), parent=mb):
                seq_inp = ValidX[i, :, :].to(device)
                seq_true = Validy[i, :, :].to(device)

                seq_pred = model(seq_inp, seq_inp[seq_length - 1:seq_length, :])

                loss = 10 * criterion_continuous(seq_pred[:, 17:24], seq_true[:, 17:24]) + criterion_binary(
                    seq_pred[:, :14], seq_true[:, :14]) + criterion_binary(seq_pred[:, 14:17], seq_true[:, 14:17])
                val_losses.append(loss.item())

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)

        history['train'].append(train_loss)
        history['val'].append(val_loss)

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), save_model_path + "_loss_val=" + str(val_loss) + ".pt")
            print("saved best model epoch:", epoch, "val loss is:", val_loss)

        print(f'Epoch {epoch}: train loss {train_loss} val loss {val_loss}')
        scheduler.step()

        plot_loss_update(epoch, history['train'], history['val'])
    # model.load_state_dict(best_model_wts)
    return model.eval(), history


def train_model_features_r_transformer(model, TrainX, Trainy, ValidX, Validy, n_epochs, device, criterion_continuous,
                                       criterion_binary, optimizer, save_model_path, clip, lr):
    history = dict(train=[], val=[], acc_train_summary=[], acc_val_summary=[], acc_train_precip_type=[],
                   acc_val_precip_type=[], mse_train_continuous=[], mse_val_continuous=[])

    # best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 10000.0
    mb = master_bar(range(1, n_epochs + 1))
    acc_summary = Accuracy(num_classes=14).to(device)
    acc_precip_type = Accuracy(num_classes=3).to(device)
    mse_continuous = MeanAbsoluteError().to(device)

    writer = SummaryWriter(f"logs/" + save_model_path)
    vloss_list = []
    for epoch in mb:
        model = model.train()

        train_losses = []
        acc_train_summary_data = []
        acc_train_precip_type_data = []
        mse_train_data = []
        train_idx_list = np.arange(len(TrainX), dtype="int32")
        np.random.shuffle(train_idx_list)
        for i in progress_bar(train_idx_list, parent=mb, ):
            seq_inp = TrainX[i].to(device)
            seq_true = Trainy[i].to(device)

            optimizer.zero_grad()
            seq_pred = model(seq_inp.unsqueeze(0)).squeeze(0)

            loss = 10 * criterion_continuous(seq_pred[:, 17:24], seq_true[:, 17:24].double()) + criterion_binary(
                seq_pred[:, :14], seq_true[:, :14].double()) + criterion_binary(seq_pred[:, 14:17],
                                                                                seq_true[:, 14:17].double())

            train_losses.append(loss.item())
            acc_train_summary_data.append(
                acc_summary(seq_pred[:, :14], seq_true[:, :14].type(torch.IntTensor).to(device)).item())
            acc_train_precip_type_data.append(
                acc_precip_type(seq_pred[:, 14:17], seq_true[:, 14:17].type(torch.IntTensor).to(device)).item())
            mse_train_data.append(mse_continuous(seq_pred[:, 17:24], seq_true[:, 17:24]).item())

            if clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            loss.backward()
            optimizer.step()

        val_losses = []
        acc_val_summary_data = []
        acc_val_precip_type_data = []
        mse_val_data = []
        model = model.eval()
        with torch.no_grad():
            valid_idx_list = np.arange(len(ValidX), dtype="int32")
            for i in progress_bar(valid_idx_list, parent=mb):
                seq_inp = ValidX[i].to(device)
                seq_true = Validy[i].to(device)

                seq_pred = model(seq_inp.unsqueeze(0)).squeeze(0)

                loss = 10 * criterion_continuous(seq_pred[:, 17:24], seq_true[:, 17:24].double()) + criterion_binary(
                    seq_pred[:, :14], seq_true[:, :14].double()) + criterion_binary(seq_pred[:, 14:17],
                                                                                    seq_true[:, 14:17].double())

                val_losses.append(loss.item())
                acc_val_summary_data.append(
                    acc_summary(seq_pred[:, :14], seq_true[:, :14].type(torch.IntTensor).to(device)).item())
                acc_val_precip_type_data.append(
                    acc_precip_type(seq_pred[:, 14:17], seq_true[:, 14:17].type(torch.IntTensor).to(device)).item())
                mse_val_data.append(mse_continuous(seq_pred[:, 17:24], seq_true[:, 17:24]).item())

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        acc_summary_val = np.mean(acc_val_summary_data)
        acc_precip_type_val = np.mean(acc_val_precip_type_data)
        acc_summary_train = np.mean(acc_train_summary_data)
        acc_precip_type_train = np.mean(acc_train_precip_type_data)
        mse_continuous_train = np.mean(mse_train_data)
        mse_continuous_val = np.mean(mse_val_data)

        history['train'].append(train_loss)
        history['val'].append(val_loss)
        history['acc_val_summary'].append(acc_summary_val)
        history['acc_val_precip_type'].append(acc_precip_type_val)
        history['acc_train_summary'].append(acc_summary_train)
        history['acc_train_precip_type'].append(acc_precip_type_train)
        history['mse_train_continuous'].append(mse_continuous_train)
        history['mse_val_continuous'].append(mse_continuous_val)

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), save_model_path + "_loss_val=" + str(val_loss) + ".pt")
            print("saved best model epoch:", epoch, "val loss is:", val_loss)

        if epoch > 10 and val_loss > max(vloss_list[-3:]):
            lr /= 10
            print('lr = {}'.format(lr) + " best_vloss=" + str(best_loss))
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        vloss_list.append(val_loss)

        print(f'Epoch {epoch}: train loss {train_loss} val loss {val_loss}')

        writer.add_scalars('Loss', {'train': train_loss, 'val': val_loss}, epoch)

        writer.add_scalars('accuracy summary', {'train': acc_summary_train, 'val': acc_summary_val}, epoch)

        writer.add_scalars('accuracy precip type', {'train': acc_precip_type_train, 'val': acc_precip_type_val}, epoch)

        writer.add_scalars('mse continuous', {'train': mse_continuous_train, 'val': mse_continuous_val}, epoch)

        plot_loss_update(epoch, history['train'], history['val'], history['acc_val_summary'],
                         history['acc_val_precip_type'], history['acc_train_summary'], history['acc_train_precip_type'],
                         history['mse_train_continuous'], history['mse_val_continuous'], save_model_path)

    # model.load_state_dict(best_model_wts)
    return model.eval(), history


def train_model_features_GRU_LSTM(model, TrainX, Trainy, Train_features, ValidX, Validy, Valid_features, seq_length,
                                  n_epochs, device, criterion_continuous, criterion_binary, optimizer, scheduler,
                                  save_model_path):
    history = dict(train=[], val=[], acc_train_summary=[], acc_val_summary=[], acc_train_precip_type=[],
                   acc_val_precip_type=[], mse_train_continuous=[], mse_val_continuous=[])

    # best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 10000.0
    mb = master_bar(range(1, n_epochs + 1))
    acc_summary = Accuracy(num_classes=14).to(device)
    acc_precip_type = Accuracy(num_classes=3).to(device)
    mse_continuous = MeanAbsoluteError().to(device)

    writer = SummaryWriter(f"logs/" + save_model_path)

    for epoch in mb:
        model = model.train()

        train_losses = []
        acc_train_summary_data = []
        acc_train_precip_type_data = []
        mse_train_data = []
        for i in progress_bar(range(TrainX.size()[0]), parent=mb, ):
            seq_inp = TrainX[i, :, :].to(device)
            seq_true = Trainy[i, :, :].to(device)
            features = Train_features[i, :, :].to(device)

            optimizer.zero_grad()
            seq_pred = model(seq_inp, seq_inp[seq_length - 1:seq_length, :], features)

            loss = 10 * criterion_continuous(seq_pred[:, 17:24], seq_true[:, 17:24]) + criterion_binary(
                seq_pred[:, :14], seq_true[:, :14]) + criterion_binary(seq_pred[:, 14:17], seq_true[:, 14:17])
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()

            train_losses.append(loss.item())
            acc_train_summary_data.append(
                acc_summary(seq_pred[:, :14], seq_true[:, :14].type(torch.IntTensor).to(device)).item())
            acc_train_precip_type_data.append(
                acc_precip_type(seq_pred[:, 14:17], seq_true[:, 14:17].type(torch.IntTensor).to(device)).item())
            mse_train_data.append(mse_continuous(seq_pred[:, 17:24], seq_true[:, 17:24]).item())

        val_losses = []
        acc_val_summary_data = []
        acc_val_precip_type_data = []
        mse_val_data = []
        model = model.eval()
        with torch.no_grad():
            for i in progress_bar(range(ValidX.size()[0]), parent=mb):
                seq_inp = ValidX[i, :, :].to(device)
                seq_true = Validy[i, :, :].to(device)
                features = Valid_features[i, :, :].to(device)

                seq_pred = model(seq_inp, seq_inp[seq_length - 1:seq_length, :], features)

                loss = 10 * criterion_continuous(seq_pred[:, 17:24], seq_true[:, 17:24]) + criterion_binary(
                    seq_pred[:, :14], seq_true[:, :14]) + criterion_binary(seq_pred[:, 14:17], seq_true[:, 14:17])
                val_losses.append(loss.item())

                acc_val_summary_data.append(
                    acc_summary(seq_pred[:, :14], seq_true[:, :14].type(torch.IntTensor).to(device)).item())
                acc_val_precip_type_data.append(
                    acc_precip_type(seq_pred[:, 14:17], seq_true[:, 14:17].type(torch.IntTensor).to(device)).item())
                mse_val_data.append(mse_continuous(seq_pred[:, 17:24], seq_true[:, 17:24]).item())

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        acc_summary_val = np.mean(acc_val_summary_data)
        acc_precip_type_val = np.mean(acc_val_precip_type_data)
        acc_summary_train = np.mean(acc_train_summary_data)
        acc_precip_type_train = np.mean(acc_train_precip_type_data)
        mse_continuous_train = np.mean(mse_train_data)
        mse_continuous_val = np.mean(mse_val_data)

        history['train'].append(train_loss)
        history['val'].append(val_loss)
        history['acc_val_summary'].append(acc_summary_val)
        history['acc_val_precip_type'].append(acc_precip_type_val)
        history['acc_train_summary'].append(acc_summary_train)
        history['acc_train_precip_type'].append(acc_precip_type_train)
        history['mse_train_continuous'].append(mse_continuous_train)
        history['mse_val_continuous'].append(mse_continuous_val)

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), save_model_path + "_loss_val=" + str(val_loss) + ".pt")
            print("saved best model epoch:", epoch, "val loss is:", val_loss)

        print(f'Epoch {epoch}: train loss {train_loss} val loss {val_loss}')
        scheduler.step()

        writer.add_scalars('Loss', {'train': train_loss, 'val': val_loss}, epoch)

        writer.add_scalars('accuracy summary', {'train': acc_summary_train, 'val': acc_summary_val}, epoch)

        writer.add_scalars('accuracy precip type', {'train': acc_precip_type_train, 'val': acc_precip_type_val}, epoch)

        writer.add_scalars('mse continuous', {'train': mse_continuous_train, 'val': mse_continuous_val}, epoch)

        plot_loss_update(epoch, history['train'], history['val'], history['acc_val_summary'],
                         history['acc_val_precip_type'], history['acc_train_summary'], history['acc_train_precip_type'],
                         history['mse_train_continuous'], history['mse_val_continuous'], save_model_path)

    # model.load_state_dict(best_model_wts)
    return model.eval(), history


def predict_example_GRU_LSTM(model, device, scaler, criterion_continuous, criterion_binary, df, TestX, Testy,
                             Test_features, in_seq_length, out_seq_length):
    acc_summary = Accuracy(num_classes=14).to(device)
    acc_precip_type = Accuracy(num_classes=3).to(device)
    mse_continuous = MeanAbsoluteError().to(device)

    test_losses = []
    acc_test_summary_data = []
    acc_test_precip_type_data = []
    mse_test_data = []
    with torch.no_grad():
        for i in range(TestX.size()[0]):
            seq_inp = TestX[i, :, :].to(device)
            seq_true = Testy[i, :, :].to(device)
            features = Test_features[i, :, :].to(device)

            seq_pred = model(seq_inp, seq_inp[in_seq_length - 1:in_seq_length, :], features)

            loss = 10 * criterion_continuous(seq_pred[:, 17:24], seq_true[:, 17:24]) + criterion_binary(
                seq_pred[:, :14], seq_true[:, :14]) + criterion_binary(seq_pred[:, 14:17], seq_true[:, 14:17])
            test_losses.append(loss.item())

            acc_test_summary_data.append(
                acc_summary(seq_pred[:, :14], seq_true[:, :14].type(torch.IntTensor).to(device)).item())
            acc_test_precip_type_data.append(
                acc_precip_type(seq_pred[:, 14:17], seq_true[:, 14:17].type(torch.IntTensor).to(device)).item())
            mse_test_data.append(mse_continuous(seq_pred[:, 17:24], seq_true[:, 17:24]).item())

    print_predict_comparison(seq_pred, seq_true, df, scaler, out_seq_length)
    test_loss = np.mean(test_losses)
    print("test loss =", test_loss)
    test_acc_test_summary = np.mean(acc_test_summary_data)
    print("test accuracy summary =", test_acc_test_summary)
    test_acc_test_precip_type = np.mean(acc_test_precip_type_data)
    print("test accuracy precip type =", test_acc_test_precip_type)
    test_mse = np.mean(mse_test_data)
    print("test mse continuous variables =", test_mse)


def predict_example_r_transformer(model, device, scaler, criterion_continuous, criterion_binary, df, TestX, Testy,
                                  out_seq_length):
    acc_summary = Accuracy(num_classes=14).to(device)
    acc_precip_type = Accuracy(num_classes=3).to(device)
    mse_continuous = MeanAbsoluteError().to(device)

    test_losses = []
    acc_test_summary_data = []
    acc_test_precip_type_data = []
    mse_test_data = []
    with torch.no_grad():
        test_idx_list = np.arange(len(TestX), dtype="int32")

        for i in test_idx_list:
            seq_inp = TestX[i].to(device)
            seq_true = Testy[i].to(device)

            seq_pred = model(seq_inp.unsqueeze(0)).squeeze(0)

            loss = 10 * criterion_continuous(seq_pred[:, 17:24], seq_true[:, 17:24].double()) + criterion_binary(
                seq_pred[:, :14], seq_true[:, :14].double()) + criterion_binary(seq_pred[:, 14:17],
                                                                                seq_true[:, 14:17].double())

            test_losses.append(loss.item())
            acc_test_summary_data.append(
                acc_summary(seq_pred[:, :14], seq_true[:, :14].type(torch.IntTensor).to(device)).item())
            acc_test_precip_type_data.append(
                acc_precip_type(seq_pred[:, 14:17], seq_true[:, 14:17].type(torch.IntTensor).to(device)).item())
            mse_test_data.append(mse_continuous(seq_pred[:, 17:24], seq_true[:, 17:24]).item())

    print_predict_comparison(seq_pred, seq_true, df, scaler, out_seq_length)
    test_loss = np.mean(test_losses)
    print("test loss =", test_loss)
    test_acc_test_summary = np.mean(acc_test_summary_data)
    print("test accuracy summary =", test_acc_test_summary)
    test_acc_test_precip_type = np.mean(acc_test_precip_type_data)
    print("test accuracy precip type =", test_acc_test_precip_type)
    test_mse = np.mean(mse_test_data)
    print("test mse continuous variables =", test_mse)


def print_predict_comparison(seq_pred, seq_true, df, scaler, out_seq_length):
    seq_pred = seq_pred.cpu().numpy()
    seq_pred_binary_summary = seq_pred[:, :14]
    seq_pred_binary_precip_type = seq_pred[:, 14:17]
    seq_pred_continuous = seq_pred[:, 17:24]

    seq_true = seq_true.cpu().numpy()
    seq_true_binary_summary = seq_true[:, :14]
    seq_true_binary_precip_type = seq_true[:, 14:17]
    seq_true_continuous = seq_true[:, 17:24]

    seq_pred_continuous = scaler.inverse_transform(seq_pred_continuous)
    seq_true_continuous = scaler.inverse_transform(seq_true_continuous)

    for row in range(out_seq_length):
        max_summary = max(seq_pred_binary_summary[row])
        seq_pred_binary_summary[row] = np.where(seq_pred_binary_summary[row] == max_summary, 1, 0)
        max_precip_type = max(seq_pred_binary_precip_type[row])
        seq_pred_binary_precip_type[row] = np.where(seq_pred_binary_precip_type[row] == max_precip_type, 1, 0)

    seq_pred_binary_summary = pd.DataFrame(seq_pred_binary_summary)
    seq_pred_binary_precip_type = pd.DataFrame(seq_pred_binary_precip_type)
    seq_pred_continuous = pd.DataFrame(seq_pred_continuous)

    seq_true_binary_summary = pd.DataFrame(seq_true_binary_summary)
    seq_true_binary_precip_type = pd.DataFrame(seq_true_binary_precip_type)
    seq_true_continuous = pd.DataFrame(seq_true_continuous)

    seq_pred_binary_summary.columns = df.columns.values.tolist()[:14]
    seq_pred_binary_precip_type.columns = df.columns.values.tolist()[14:17]
    seq_pred_continuous.columns = df.columns.values.tolist()[17:24]

    seq_true_binary_summary.columns = df.columns.values.tolist()[:14]
    seq_true_binary_precip_type.columns = df.columns.values.tolist()[14:17]
    seq_true_continuous.columns = df.columns.values.tolist()[17:24]

    for column_label in df.columns.values.tolist()[17:24]:
        plt.plot(seq_true_continuous[column_label], label='true')
        plt.plot(seq_pred_continuous[column_label], label='predict')
        plt.title(column_label)
        plt.legend()
        plt.show()

    plt.plot(seq_true_binary_precip_type.idxmax(axis=1), label='true')
    plt.plot(seq_pred_binary_precip_type.idxmax(axis=1), label='predict')
    plt.title("Precip type")
    plt.legend()
    plt.show()

    plt.plot(seq_pred_binary_summary.idxmax(axis=1), label='true')
    plt.plot(seq_true_binary_summary.idxmax(axis=1), label='predict')
    plt.title("Summary")
    plt.legend()
    plt.show()


def sliding_windows(data, in_seq_length, out_seq_length):
    x = []
    y = []

    for i in range(len(data) - (in_seq_length + out_seq_length)):
        _x = data[i:(i + in_seq_length)]
        _y = data[(i + in_seq_length):(i + in_seq_length + out_seq_length)]
        x.append(_x)
        y.append(_y)

    return x, y


def sliding_windows_features(data, in_seq_length, out_seq_length):
    x = []
    y = []
    z = []

    for i in range(len(data) - (in_seq_length + out_seq_length)):
        _x = data.iloc[i:(i + in_seq_length), :]
        _y = data.iloc[(i + in_seq_length):(i + in_seq_length + out_seq_length), :24]
        _z = data.iloc[(i + in_seq_length):(i + in_seq_length + out_seq_length), 24:]
        x.append(np.array(_x))
        y.append(np.array(_y))
        z.append(np.array(_z))

    return x, y, z


def weather_to_r_transformer_shape(data):
    data_rtransformer = np.empty((len(data)), np.object)

    for i in range(len(data)):
        tensor = torch.Tensor(data[i, :, :].astype(np.float64))
        data_rtransformer[i] = tensor

    return data_rtransformer


def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)
