import torch
import torch.nn as nn
from torch.autograd import Variable


class Encoder(nn.Module):
    def __init__(self, seq_len, n_features, device, embedding_dim, num_layers, dropout, bidirectional=False):
        super(Encoder, self).__init__()

        self.seq_len = seq_len
        self.n_features = n_features
        self.hidden_size = embedding_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional

        self.rnn1 = nn.LSTM(input_size=n_features, hidden_size=self.hidden_size, num_layers=self.num_layers,
            batch_first=True, dropout=self.dropout, bidirectional=self.bidirectional)
        self.device = device

    def forward(self, x):
        x = x.reshape((1, self.seq_len, self.n_features))

        h_1 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device))

        c_1 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device))

        x, (hidden, cell) = self.rnn1(x, (h_1, c_1))

        return hidden, cell


class Decoder(nn.Module):
    def __init__(self, seq_len, input_dim, n_features, num_layers, dropout, bidirectional=False):
        super(Decoder, self).__init__()

        self.seq_len = seq_len
        self.hidden_dim = input_dim
        self.n_features = n_features
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional

        self.rnn1 = nn.LSTM(input_size=self.n_features, hidden_size=self.hidden_dim, num_layers=self.num_layers,
            batch_first=True, dropout=self.dropout, bidirectional=bidirectional)

        self.output_layer = nn.Linear(self.hidden_dim, self.n_features)

    def forward(self, x, input_hidden, input_cell):
        x = x.reshape((1, 1, self.n_features))
        x, (hidden_n, cell_n) = self.rnn1(x, (input_hidden, input_cell))

        x = self.output_layer(x)
        return x, hidden_n, cell_n


class Seq2SeqLSTM(nn.Module):

    def __init__(self, seq_len, n_features, device, output_length, embedding_dim, encoder_num_layers=3,
                 encoder_dropout=0.35, encoder_bidirectional=False, decoder_num_layers=3, decoder_dropout=0.35,
                 decoder_bidirectional=False):
        super(Seq2SeqLSTM, self).__init__()

        self.encoder = Encoder(seq_len, n_features, device, embedding_dim, encoder_num_layers, encoder_dropout,
                               encoder_bidirectional).to(device)
        self.n_features = n_features
        self.output_length = output_length
        self.decoder = Decoder(seq_len, embedding_dim, n_features, decoder_num_layers, decoder_dropout,
                               decoder_bidirectional).to(device)

    def forward(self, x, prev_y, features):
        hidden, cell = self.encoder(x)
        targets_ta = []
        dec_input = prev_y

        for out_days in range(self.output_length):
            prev_x, prev_hidden, prev_cell = self.decoder(dec_input, hidden, cell)
            hidden, cell = prev_hidden, prev_cell
            prev_x = prev_x[:, :, :24]

            if out_days + 1 < self.output_length:
                dec_input = torch.cat([prev_x, features[out_days + 1].reshape(1, 1, 102)], dim=2)

            targets_ta.append(prev_x.reshape(24))
        targets = torch.stack(targets_ta)

        return targets
