import torch
from torch.utils.data import Dataset
from collections import OrderedDict


class CustomDataset(Dataset):
    def __init__(self, df, numerical_columns, categorical_columns, target):
        self.numerical_columns = numerical_columns
        self.categorical_columns = categorical_columns
        self.X = df[numerical_columns+categorical_columns]
        self.y = df[target]

        self.num_unique_values_dict = {}
        for col in self.categorical_columns:
            self.num_unique_values_dict[col] = len(self.X[col].unique())

    def __len__(self):
        return len(self.X)

    def __getitem__(self, indx):
        row = self.X.iloc[indx, :]
        numerical_values = row[self.numerical_columns].values
        category_values = OrderedDict({col: torch.tensor(row[col], dtype=torch.long)
                                       for col in self.categorical_columns})
        target = self.y[indx]
        return category_values, numerical_values, target
