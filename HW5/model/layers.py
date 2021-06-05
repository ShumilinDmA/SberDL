import torch
import torch.nn as nn
import torch.nn.functional as f
from model.utils import GBN, split, Sparsemax


class DenseFeatureLayer(nn.Module):
    def __init__(self, num_unique_values_dict: dict, embedding_size: int, n_numerical_columns: int):
        super(DenseFeatureLayer, self).__init__()
        self.num_unique_values_dict = num_unique_values_dict
        self.embedding_size = embedding_size

        self.embedding_layers = nn.ModuleDict({col: nn.Embedding(num, self.embedding_size) for col, num in
                                               self.num_unique_values_dict.items()})
        self.output_shape = len(self.num_unique_values_dict) * embedding_size + n_numerical_columns
        self.batch_norm = nn.BatchNorm1d(self.output_shape)

    def forward(self, categorical, numerical):
        x_cat = [self.embedding_layers[col](number) for col, number in categorical.items()]
        x_cat = torch.cat(x_cat, dim=-1)
        features = torch.cat((x_cat, numerical), dim=-1).float()
        output = self.batch_norm(features)
        return output


class TransformerBlock(nn.Module):
    def __init__(self, input_size, output_size, skip=True, virtual_batch_size=128, momentum=0.98):
        super(TransformerBlock, self).__init__()
        self.skip = skip
        self.fc = nn.Linear(input_size, output_size*2)
        self.batch_norm = GBN(output_size*2, virtual_batch_size=virtual_batch_size, momentum=momentum)

    def forward(self, data):
        x = self.fc(data)
        x = self.batch_norm(x)
        x = f.glu(x)
        if self.skip:
            x = x + torch.sqrt(torch.tensor(0.5))*data
        return x


class SharedTransformerBlock(nn.Module):
    def __init__(self, num_shared_layers: int,  input_size: int, output_size: int,
                 virtual_batch_size: int = 128, momentum: float = 0.98):
        super(SharedTransformerBlock, self).__init__()

        self.input_size = input_size
        self.output_size = output_size

        self.layers = nn.ModuleList([
            TransformerBlock(input_size=self.input_size, output_size=self.output_size, skip=False,
                             virtual_batch_size=virtual_batch_size, momentum=momentum) if i == 0 else
            TransformerBlock(input_size=self.output_size, output_size=self.output_size,
                             virtual_batch_size=virtual_batch_size, momentum=momentum)
            for i in range(num_shared_layers)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class FeatureTransformer(nn.Module):
    def __init__(self, n_decision_blocks: int, input_size: int = None, output_size: int = None, shared_block=None,
                 virtual_batch_size: int = 128, momentum: float = 0.98):
        super(FeatureTransformer, self).__init__()
        self.shared_block = shared_block
        self.n_decision_block = n_decision_blocks
        self.input_size = input_size
        self.output_size = output_size

        if self.shared_block:
            self.hidden_size = self.shared_block.output_size

            self.decision_layers = nn.ModuleList([
                TransformerBlock(input_size=self.hidden_size, output_size=self.hidden_size,
                                 virtual_batch_size=virtual_batch_size, momentum=momentum)
                for _ in range(n_decision_blocks)
            ])

        else:
            self.decision_layers = nn.ModuleList([
                TransformerBlock(input_size=self.input_size, output_size=self.output_size, skip=False,
                                 virtual_batch_size=virtual_batch_size, momentum=momentum) if i == 0 else
                TransformerBlock(input_size=self.output_size, output_size=self.output_size,
                                 virtual_batch_size=virtual_batch_size, momentum=momentum)
                for i in range(n_decision_blocks)])

    def forward(self, x):
        if self.shared_block:
            x = self.shared_block(x)
            for layer in self.decision_layers:
                x = layer(x)
            return x
        else:
            for layer in self.decision_layers:
                x = layer(x)
            return x


class AttentiveTransformer(nn.Module):
    def __init__(self, input_size: int, output_size: int,
                 virtual_batch_size: int, momentum: float):
        super(AttentiveTransformer, self).__init__()
        self.virtual_batch_size = virtual_batch_size
        self.momentum = momentum
        self.input_size = input_size
        self.output_size = output_size
        self.fc = nn.Linear(self.input_size, self.output_size)
        self.batch_norm = GBN(self.output_size, virtual_batch_size=self.virtual_batch_size, momentum=self.momentum)
        self.sparcemax = Sparsemax(dim=-1)

    def forward(self, batch, prior_scales):
        x = self.fc(batch)
        x = self.batch_norm(x)
        x = prior_scales * x
        mask = self.sparcemax(x)
        return mask


class DecisionStepBlock(nn.Module):
    def __init__(self, shared_block, meaningful_part: float, n_decision_block: int,
                 virtual_batch_size: int, momentum: float):
        super(DecisionStepBlock, self).__init__()
        self.shared_block = shared_block
        self.n_decision_block = n_decision_block
        self.meaningful_part = meaningful_part
        self.virtual_batch_size = virtual_batch_size
        self.momentum = momentum
        self.attention_input = self.shared_block.output_size - int(self.meaningful_part * self.shared_block.output_size)
        self.attention_output = self.shared_block.input_size

        self.AttentiveTransformer = AttentiveTransformer(input_size=self.attention_input,
                                                         output_size=self.attention_output,
                                                         virtual_batch_size=self.virtual_batch_size,
                                                         momentum=self.momentum)

        self.FeatureTransformer = FeatureTransformer(n_decision_blocks=self.n_decision_block,
                                                     shared_block=self.shared_block,
                                                     virtual_batch_size=self.virtual_batch_size,
                                                     momentum=self.momentum)

    def forward(self, input_data, to_attention, features, prior_scales):
        mask = self.AttentiveTransformer(to_attention, prior_scales)
        x = mask * features
        x = self.FeatureTransformer(x)
        left_data, right_data = split(x, meaningful_part=self.meaningful_part)
        output_data = f.relu(left_data) + input_data
        return output_data, right_data,  mask, features
