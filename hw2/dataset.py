import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler

class MLDataset(Dataset):
    def __init__(self, data):

        # label's columns name, no need to rewrite
        label_col = [
            'Input_A6_024' ,'Input_A3_016', 'Input_C_013', 'Input_A2_016', 'Input_A3_017',
            'Input_C_050', 'Input_A6_001', 'Input_C_096', 'Input_A3_018', 'Input_A6_019',
            'Input_A1_020', 'Input_A6_011', 'Input_A3_015', 'Input_C_046', 'Input_C_049',
            'Input_A2_024', 'Input_C_058', 'Input_C_057', 'Input_A3_013', 'Input_A2_017'
        ]
        # ================================================================================ #
        # Do any operation on self.train you want with data type "dataframe"(recommanded) in this block.
        # For example, do normalization or dimension Reduction.
        # Some of columns have "nan", need to drop row or fill with value first
        # For example:

        # fill NAN with the median values
        data = data.fillna(value=data.median(axis=0, skipna=True))

        # normalize with MinMaxScaler in range [0, 1]
        scaler = MinMaxScaler(feature_range=(0, 1), copy=True).fit(data)
        data = pd.DataFrame(data=scaler.transform(data), columns=data.columns)

        self.label = data[label_col]
        self.data = data.drop(label_col, axis=1)

        # ================================================================================ #

    def __len__(self):
        #  no need to rewrite
        return len(self.data)

    def __getitem__(self, index):
        # transform dataframe to numpy array, no need to rewrite
        x = self.data.iloc[index, :].values
        y = self.label.iloc[index, :].values
        return x, y

