import torch.nn as nn
from model.tabnet import TabNet


class ClassificationBlock(nn.Module):
    def __init__(self, input_size: int, output_size: int):
        super(ClassificationBlock, self).__init__()

        self.fc = nn.Linear(input_size, output_size)
        self.batch_norm = nn.BatchNorm1d(num_features=output_size)

    def forward(self, batch):
        x = self.fc(batch)
        x = self.batch_norm(x)
        return x


class TabNetClassifier(nn.Module):
    def __init__(self, n_output_classes: int,
                 n_classification_layer: int,
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
        super(TabNetClassifier, self).__init__()

        self.n_output_classes = n_output_classes
        self.n_classification_layer = n_classification_layer
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

        self.TabNet = TabNet(num_unique_values_dict=self.num_unique_values_dict,
                             embedding_size=self.embedding_size,
                             n_numerical_columns=self.n_numerical_columns,
                             n_shared_layers=self.n_shared_layers,
                             n_decision_blocks=self.n_decision_blocks,
                             hidden_size=self.hidden_size,
                             meaningful_part=self.meaningful_part,
                             n_decision_steps=self.n_decision_steps,
                             virtual_batch_size=self.virtual_batch_size,
                             momentum=self.momentum,
                             gamma=self.gamma)

        self.classification_layers = nn.ModuleList([ClassificationBlock(input_size=self.TabNet.tabnet_output_size,
                                                                        output_size=self.TabNet.tabnet_output_size)
                                                    if i != (self.n_classification_layer-1) else
                                                    ClassificationBlock(input_size=self.TabNet.tabnet_output_size,
                                                                        output_size=self.n_output_classes)
                                                    for i in range(self.n_classification_layer)])

        self.batch_norm = nn.BatchNorm1d(self.TabNet.tabnet_output_size)

    def forward(self, batch):
        x, masks = self.TabNet(batch)
        x = self.batch_norm(x)
        for layer in self.classification_layers:
            x = layer(x)
        return x, masks
