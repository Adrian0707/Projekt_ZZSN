import torch
import torch.nn as nn
import torch.nn.functional as F
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

        return x, hidden, cell


class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()

        self.attn = nn.Linear((enc_hid_dim) + dec_hid_dim, dec_hid_dim)
        self.v = nn.Linear(dec_hid_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        src_len = encoder_outputs.shape[1]
        hidden = hidden[2:3, :, :]
        hidden = hidden.repeat(1, src_len, 1)

        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        attention = self.v(energy).squeeze(2)

        return F.softmax(attention, dim=1)


class AttentionDecoder(nn.Module):
    def __init__(self, seq_len, attention, input_dim, n_features, num_layers, dropout, bidirectional=False):
        super(AttentionDecoder, self).__init__()

        self.seq_len = seq_len
        self.hidden_dim = input_dim
        self.n_features = n_features
        self.attention = attention
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional

        self.rnn1 = nn.LSTM(input_size=self.hidden_dim + self.n_features, hidden_size=self.hidden_dim,
            num_layers=self.num_layers, batch_first=True, dropout=self.dropout, bidirectional=bidirectional)

        self.output_layer = nn.Linear(self.hidden_dim * 2, n_features)

    def forward(self, x, input_hidden, input_cell, encoder_outputs):
        a = self.attention(input_hidden, encoder_outputs)
        a = a.unsqueeze(1)

        weighted = torch.bmm(a, encoder_outputs)

        x = x.reshape((1, 1, self.n_features))

        rnn_input = torch.cat((x, weighted), dim=2)

        x, (hidden_n, cell_n) = self.rnn1(rnn_input, (input_hidden, input_cell))

        output = x.squeeze(0)
        weighted = weighted.squeeze(0)

        x = self.output_layer(torch.cat((output, weighted), dim=1))
        return x, hidden_n, cell_n


class Seq2Seq(nn.Module):

    def __init__(self, seq_len, n_features, device, output_length, embedding_dim=512, encoder_num_layers=3,
                 encoder_dropout=0.35, encoder_bidirectional=False, decoder_num_layers=3, decoder_dropout=0.35,
                 decoder_bidirectional=False):
        super(Seq2Seq, self).__init__()

        self.encoder = Encoder(seq_len, n_features, device, embedding_dim=embedding_dim, num_layers=encoder_num_layers,
                               dropout=encoder_dropout, bidirectional=encoder_bidirectional).to(device)

        self.attention = Attention(embedding_dim, embedding_dim)

        self.decoder = AttentionDecoder(seq_len, self.attention, input_dim=embedding_dim, n_features=n_features,
                                        num_layers=decoder_num_layers, dropout=decoder_dropout,
                                        bidirectional=decoder_bidirectional).to(device)

        self.output_length = output_length

        self.n_features = n_features

    def forward(self, x, prev_y):
        encoder_output, hidden, cell = self.encoder(x)
        targets_ta = []
        prev_output = prev_y

        for out_days in range(self.output_length):
            prev_x, prev_hidden, prev_cell = self.decoder(prev_output, hidden, cell, encoder_output)
            hidden, cell = prev_hidden, prev_cell
            prev_output = prev_x
            targets_ta.append(prev_x.reshape(self.n_features))
        targets = torch.stack(targets_ta)

        return targets
