import torch
import torch.nn as nn
import torch.nn.functional as f
from model.utils import GBN


class DenseFeatureLayer(nn.Module):
    def __init__(self, num_unique_values_dict: dict, embedding_size: int, numerical_columns: int):
        super(DenseFeatureLayer, self).__init__()
        self.num_unique_values_dict = num_unique_values_dict
        self.embedding_size = embedding_size

        self.embedding_layers = nn.ModuleDict({col: nn.Embedding(num, self.embedding_size) for col, num in
                                               self.num_unique_values_dict.items()})
        self.output_shape = len(self.num_unique_values_dict) * embedding_size + numerical_columns
        self.batch_norm = nn.BatchNorm1d(self.output_shape)

    def forward(self, categorical, numerical):
        x_cat = [self.embedding_layers[col](number) for col, number in categorical.items()]
        x_cat = torch.cat(x_cat, dim=-1)
        features = torch.cat((x_cat, numerical), dim=-1).float()
        output = self.batch_norm(features)
        return output


class TransformerBlock(nn.Module):
    def __init__(self, input_size, skip=True, virtual_batch_size=128, momentum=0.98):
        super(TransformerBlock, self).__init__()
        self.skip = skip
        self.fc = nn.Linear(input_size, input_size*2)
        self.batch_norm = GBN(input_size*2, virtual_batch_size=virtual_batch_size, momentum=momentum)

    def forward(self, data):
        x = self.fc(data)
        x = self.batch_norm(x)
        x = f.glu(x)
        if self.skip:
            x = x + torch.sqrt(torch.tensor(0.5))*data
        return x
