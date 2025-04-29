import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd

from torch.utils.data import Dataset


np.random.seed(42)
torch.manual_seed(1)

def generate_simple_synthetic_data(n_samples=1000, seed=42):
        np.random.seed(seed)

        # Contextual features
        x1 = np.random.uniform(0, 1, size=n_samples)   # e.g., income level normalized [0, 1]
        x2 = np.random.uniform(0, 1, size=n_samples)   # e.g., brand loyalty normalized [0, 1]

        # Price feature (higher price usually decreases demand)
        price = np.random.uniform(1, 10, size=n_samples)  # price between $1 and $10

        # Define true mean (nonlinear relationship)
        true_mean = (5 * x1 + 3 * x2) / (1 + 0.5 * price)  # demand decreases with higher price

        # Define true std (slightly depends on features)
        true_std = 0.2 + 0.3 * np.abs(x1 - x2) + 0.05 * (price / 10)  # more price variation adds more noise

        # Sample y from N(mean, std)
        y = np.random.normal(loc=true_mean, scale=true_std)

        # Package into dataframe
        X = pd.DataFrame({'x1': x1, 'x2': x2, 'price': price})
        y = pd.Series(y, name='y')

        return X, y, true_mean, true_std

def extract_layer(weight, bias, l):
        df_sub = pd.DataFrame(weight[l]).add_prefix('node_')
        df_sub['intercept'] = bias[l]
        df_sub['layer'] = l
        df_sub['node'] = range(len(df_sub))
        return df_sub

def constraint_extrapolation_MLP(weight, bias, names):
    n_layers = len(names)
    constraints = pd.concat([extract_layer(weight, bias, l) for l in range(n_layers)],axis=0)
    cols_to_move = ['intercept', 'layer', 'node']
    constraints = constraints[cols_to_move + [col for col in constraints.columns if col not in cols_to_move]]
    return constraints

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class PandasDataset(Dataset):
    def __init__(self, X, y=None, transform=None):
        self.X = X.values if isinstance(X, pd.DataFrame) else X
        self.y = y.values if isinstance(y, pd.Series) else y
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = torch.tensor(self.X[idx], dtype=torch.float32)
        if self.transform:
            x = self.transform(x)
        if self.y is not None:
            y = torch.tensor(self.y[idx], dtype=torch.float32)
            return x, y
        return x


