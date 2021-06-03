import torch
import torch.nn as nn

from model.layers import DenseFeatureLayer, FeatureTransformer, SharedTransformerBlock, DecisionStepBlock
from model.utils import split


class TabNet(nn.Module):
    def __init__(self,
                 num_unique_values_dict: dict,
                 embedding_size: int,
                 n_numerical_columns: int,
                 n_shared_layers: int,
                 n_decision_blocks: int,
                 hidden_size: int,
                 meaningful_part: float,
                 n_decision_steps: int,
                 virtual_batch_size: int,
                 momentum: float,
                 gamma: float
                 ):

        super(TabNet, self).__init__()

        self.num_unique_values_dict = num_unique_values_dict
        self.embedding_size = embedding_size
        self.n_numerical_columns = n_numerical_columns
        self.n_shared_layers = n_shared_layers
        self.n_decision_blocks = n_decision_blocks
        self.hidden_size = hidden_size
        self.meaningful_part = meaningful_part
        self.n_decision_steps = n_decision_steps
        self.virtual_batch_size = virtual_batch_size
        self.momentum = momentum
        self.gamma = gamma

        self.feature_layer = DenseFeatureLayer(num_unique_values_dict=self.num_unique_values_dict,
                                               embedding_size=self.embedding_size,
                                               n_numerical_columns=self.n_numerical_columns)

        self.first_feature_transformer = FeatureTransformer(n_decision_blocks=self.n_decision_blocks,
                                                            input_size=self.feature_layer.output_shape,
                                                            output_size=self.hidden_size,
                                                            virtual_batch_size=self.virtual_batch_size,
                                                            momentum=self.momentum)

        self.shared_block = SharedTransformerBlock(num_shared_layers=self.n_shared_layers,
                                                   input_size=self.feature_layer.output_shape,
                                                   output_size=self.hidden_size,
                                                   virtual_batch_size=self.virtual_batch_size,
                                                   momentum=self.momentum)

        self.decision_steps_list = nn.ModuleList([
            DecisionStepBlock(shared_block=self.shared_block,
                              n_decision_block=self.n_decision_blocks,
                              virtual_batch_size=self.virtual_batch_size,
                              momentum=self.momentum,
                              meaningful_part=self.meaningful_part,
                              gamma=self.gamma)
            for _ in range(self.n_decision_steps)
        ])

        self.tabnet_output_size = int(self.meaningful_part * self.shared_block.output_size)

    def forward(self, batch):
        features = self.feature_layer(batch[0], batch[1])
        post_processed_features = self.first_feature_transformer(features)
        left_data, right_data = split(post_processed_features, self.meaningful_part)

        output_data = torch.zeros_like(left_data)

        mask_list = []
        for layer in self.decision_steps_list:
            output_data, right_data, mask, features = layer(input_data=output_data, to_attention=right_data,
                                                            features=features)
            mask_list.append(mask)
        mask_list = torch.stack(mask_list)

        return output_data, mask_list
