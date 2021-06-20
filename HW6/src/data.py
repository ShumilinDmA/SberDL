import torch
from torch.utils.data import Dataset
import numpy as np
import pickle


class CustomDataset(Dataset):
    def __init__(self, df, cat_columns, num_columns, target_columns, drop_columns, max_latest_seq=None):
        self.df = df
        self.cat_columns = cat_columns
        self.num_columns = num_columns
        self.target_columns = target_columns
        self.drop_columns = drop_columns
        self.max_latest_seq = max_latest_seq
        self.indx_to_user = {i: user for i, user in enumerate(self.df.User.unique())}

    def __len__(self):
        return len(self.indx_to_user)

    def __getitem__(self, indx):
        user_id = self.indx_to_user[indx]
        user_data = (self.df.loc[self.df.loc[:, 'User'] == user_id, :]
                     .drop(columns=self.drop_columns)
                     .reset_index(drop=True))

        if self.max_latest_seq:
            if len(user_data) > self.max_latest_seq:
                user_data = user_data[-self.max_latest_seq:]

        cat_data = user_data[self.cat_columns].to_numpy()
        num_data = user_data[self.num_columns].to_numpy()
        target = np.unique(user_data[self.target_columns].to_numpy())[0]
        return cat_data, num_data, target


def collate_fn(batch):
    category_data = []
    numerical_data = []
    target_data = []
    for category, numerical, target in batch:
        category_data.append(torch.LongTensor(category))
        numerical_data.append(torch.Tensor(numerical))
        target_data.append(target)

    category_data = torch.nn.utils.rnn.pad_sequence(category_data, batch_first=True, padding_value=137)

    target_data = torch.LongTensor(target_data)

    return category_data, numerical_data, target_data


def load_dataset(path, cfg):

    # Load from pickle
    with open(path, "rb") as file:
        df = pickle.load(file)
        df.sort_values(by='Date', inplace=True)

    # Specify columns and split on groups
    cat_columns = ["Card", "Use Chip", "MCC", "Errors", "Outcome", "is_diff_merchant",
                   "is_diff_merchant_city", "is_diff_merchant_state"]
    target_columns = ["IsFraud_target"]
    drop_columns = ["User", "Date"]
    num_columns = np.setdiff1d(df.columns.tolist(), cat_columns + target_columns + drop_columns).tolist()
    assert len(df.columns) == len(cat_columns + target_columns + drop_columns + num_columns)

    dataset = CustomDataset(df, cat_columns=cat_columns,
                            num_columns=num_columns,
                            target_columns=target_columns,
                            drop_columns=drop_columns,
                            max_latest_seq=cfg.max_seq_len_latest)
    return dataset
