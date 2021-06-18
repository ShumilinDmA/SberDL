import torch
import torch.nn as nn
import torch.nn.functional as f


class RNNnetwork(nn.Module):
    def __init__(self,
                 num_uniq_embeddings,
                 embedding_dim,
                 feature_dim,
                 n_lstm_layer,
                 hidden_dim,
                 dropout,
                 bidirectional,
                 n_numerical_col):
        super(RNNnetwork, self).__init__()

        self.num_uniq_embeddings = num_uniq_embeddings
        self.embedding_dim = embedding_dim
        self.feature_dim = feature_dim
        self.n_lstm_layer = n_lstm_layer
        self.hidden_dim = hidden_dim
        self.dropout = dropout if n_lstm_layer > 1 else 0
        self.bidirectional = bidirectional
        self.n_numerical_col = n_numerical_col

        self.embedding = nn.Embedding(self.num_uniq_embeddings, self.embedding_dim)
        self.batch_norm = nn.BatchNorm1d(self.n_numerical_col)
        self.rnn = nn.LSTM(input_size=self.feature_dim,
                           num_layers=self.n_lstm_layer,
                           hidden_size=self.hidden_dim,
                           batch_first=True,
                           dropout=self.dropout,
                           bidirectional=self.bidirectional)

        self.linear_one = nn.Linear(self.hidden_dim * (self.bidirectional + 1), self.hidden_dim)
        self.batchnorm_linear = nn.BatchNorm1d(self.hidden_dim)
        self.linear_two = nn.Linear(self.hidden_dim, 2)

    def forward(self, batch):
        cat, num = batch

        # Concat embeddings
        cat = self.embedding(cat)
        all_but_last_two_dims = cat.size()[:-2]
        cat = cat.view(*all_but_last_two_dims, -1)

        # Batchnorm across numeric features and pack it to tensor
        num = [self.batch_norm(numeric) for numeric in num]
        num = torch.nn.utils.rnn.pad_sequence(num, batch_first=True, padding_value=0)

        # Concat all features
        batch = torch.cat((num, cat), dim=-1)

        out, (hidden, cell) = self.rnn(batch)

        if self.bidirectional:
            to_classifier = torch.cat((hidden[-2], hidden[-1]), dim=-1)
        else:
            to_classifier = hidden[-1]

        x = f.relu(self.linear_one(to_classifier))
        x = self.batchnorm_linear(x)
        output = self.linear_two(x)
        return output
