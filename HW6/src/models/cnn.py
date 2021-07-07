import torch
import torch.nn as nn
import torch.nn.functional as f


class CNNnetwork(nn.Module):
    def __init__(self,
                 num_uniq_embeddings,
                 embedding_dim,
                 n_numerical_col,
                 feature_dim,
                 out_channels,
                 kernel_sizes,
                 dropout):
        super(CNNnetwork, self).__init__()

        self.num_uniq_embeddings = num_uniq_embeddings
        self.embedding_dim = embedding_dim
        self.n_numerical_col = n_numerical_col
        self.feature_dim = feature_dim
        self.out_channels = out_channels
        if isinstance(kernel_sizes, list):
            self.kernel_sizes = kernel_sizes
        elif isinstance(kernel_sizes, str):
            self.kernel_sizes = [int(x) for x in kernel_sizes.split(",")]
        else:
            raise ValueError("Should be or list of int or string. Example: 2, 3, 4, 5 ")

        self.dropout = dropout

        self.embedding = nn.Embedding(self.num_uniq_embeddings, self.embedding_dim)
        self.batch_norm = nn.BatchNorm1d(self.n_numerical_col)

        self.conv_list = nn.ModuleList([nn.Conv1d(in_channels=self.feature_dim,
                                                  out_channels=self.out_channels,
                                                  kernel_size=(kernel_size,)) for kernel_size in self.kernel_sizes])

        self.batchnorm1 = nn.BatchNorm1d(len(self.kernel_sizes) * self.out_channels)
        self.dropout_layer = nn.Dropout(self.dropout)
        self.fc1 = nn.Linear(len(self.kernel_sizes) * self.out_channels, len(self.kernel_sizes) * self.out_channels)
        self.batchnorm2 = nn.BatchNorm1d(len(self.kernel_sizes) * self.out_channels)
        self.fc2 = nn.Linear(len(self.kernel_sizes) * self.out_channels, 2)

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
        embedded = torch.cat((num, cat), dim=-1)  # [batch_size, seq_len, emb_dim]

        embedded = embedded.permute(0, 2, 1)  # [batch_size, emb_dim, seq_len]

        convoluted = [f.relu(conv(embedded)) for conv in
                      self.conv_list]  # [batch_size, out_channel, seq_len-(kernel-1)]

        pooled = [f.max_pool1d(x_conv, x_conv.shape[2]) for x_conv in convoluted]

        x = self.batchnorm1(torch.cat(pooled, dim=1).squeeze(2))
        x = self.dropout_layer(x)
        x = f.relu(self.fc1(x))
        x = self.batchnorm2(x)
        x = self.fc2(x)

        return x
