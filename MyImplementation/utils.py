import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd

from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

np.random.seed(42)
torch.manual_seed(1)

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

# Model definition
class distModel(nn.Module):
    def __init__(self, input_size, hidden_layers, hidden_size, drop):
        super().__init__() # Call the parent class (nn.Module) constructor.
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.hidden_size = hidden_size
        self.drop = drop

        # Define the input layer
        self.input_layer = nn.Linear(self.input_size, self.hidden_size)

        # Define the hidden layers
        self.hidden_layers = nn.ModuleList([nn.Linear(self.hidden_size, self.hidden_size) for _ in range(self.hidden_layers)])

        # Output mean Layer
        self.output_mean_layer = nn.Linear(self.hidden_size, 1)
        
        # Output var Layer
        self.output_sd_layer = nn.Linear(self.hidden_size, 1)

        # Define the activation function
        self.relu = nn.ReLU()

        # Define the dropout layer
        self.dropout = nn.Dropout(self.drop)

    def forward(self, x):
        # Define the forward pass
        x = self.relu(self.input_layer(x))
        for hidden_layer in self.hidden_layers:
            x = self.relu(hidden_layer(x))
            x = self.dropout(x)
        mean = self.output_mean_layer(x)
        sd = self.relu(self.output_sd_layer(x)) # Ensure variance is non-negative, can still be 0. ReLU is linear.
        #sd = self.relu(self.output_sd_layer(x)) + 1e-6 # Ensure variance is positive, can't be 0. ReLU is linear.
        #sd = self.output_sd_layer(x)
        return mean, sd

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


# Define trainer class
class Trainer():
    torch.manual_seed(1)
    def __init__(self, X, y, model_config, training_config, device='cpu'):
        self.device = device
        self.X = X
        self.y = y

        self.model_config = model_config
        self.training_config = training_config

        # Definde the model
        self.model = distModel(input_size=self.X.shape[1], hidden_layers=self.model_config['hidden_layers'], hidden_size=self.model_config['hidden_size'], drop=self.model_config['drop']).to(self.device)

        # Define criterion: https://pytorch.org/docs/stable/generated/torch.nn.GaussianNLLLoss.html
        self.criterion = nn.GaussianNLLLoss()

    def create_data_partitions(self, create_test=True, test_size=0.2, val_size=0.1, random_state=42):
        '''
        Splits self.X and self.y into train, validation, and optionally test sets.

        Parameters:
        - create_test (bool): Whether to create a test set.
        - test_size (float): Proportion of data for the test set (if created).
        - val_size (float): Proportion of the training data for validation.
        - random_state (int): Random seed for reproducibility.

        Returns:
        - Dictionaries containing the partitions.
        '''
        X_train, X_temp, y_train, y_temp = train_test_split(self.X, self.y, 
                                                            test_size=(test_size + val_size) if create_test else val_size, 
                                                            random_state=random_state)
        
        self.X_train = X_train
        self.X_val = X_temp
        self.y_train = y_train
        self.y_temp = y_temp
        
        if create_test:
            X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, 
                                                            test_size=test_size / (test_size + val_size), 
                                                            random_state=random_state)
            self.X_val = X_val
            self.X_test = X_test
            self.y_val = y_val
            self.y_test = y_test
    
    def set_data_partitions(self, X_train, y_train, X_val, y_val, X_test=None, y_test=None):
        '''
        Sets the data partitions directly.

        Parameters:
        - X_train (pd.DataFrame): Training features.
        - y_train (pd.Series): Training target.
        - X_val (pd.DataFrame): Validation features.
        - y_val (pd.Series): Validation target.
        - X_test (pd.DataFrame, optional): Test features.
        - y_test (pd.Series, optional): Test target.
        '''
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val

        if X_test is not None and y_test is not None:
            self.X_test = X_test
            self.y_test = y_test
            
    def train(self):
        torch.cuda.empty_cache()

        train_ds = PandasDataset(self.X_train, self.y_train)
        val_ds = PandasDataset(self.X_val, self.y_val)
        
        train_dl = DataLoader(train_ds, batch_size = self.training_config['batchsize'], shuffle=True)
        val_dl = DataLoader(val_ds, batch_size=len(val_ds)) # self.training_config['batchsize']?

        iters = self.training_config['iters']
        # Define optimizer: https://pytorch.org/docs/stable/generated/torch.optim.Adam.html
        optimizer = optim.Adam(self.model.parameters(), lr = self.training_config['lr'], weight_decay=self.training_config['wd'])

        best_loss = float('inf')
        for epoch in range(iters):
            ## Training
            self.model.train()
            self.train_loss = AverageMeter()
            for i, (x, y) in enumerate(train_dl):
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                mean, sd = self.model(x)
                #print("sd: ", len(sd))
                #mean, var = self.model(x)
                loss = self.criterion(mean, y, sd**2)
                #loss = self.criterion(mean, y, var)
                loss.backward()
                optimizer.step()
                self.train_loss.update(loss.item(), x.size(0))

            ## Validation
            self.model.eval()
            self.val_loss = AverageMeter()
            with torch.no_grad():
                for i, (x, y) in enumerate(val_dl):
                    x, y = x.to(self.device), y.to(self.device)
                    mean, sd = self.model(x)
                    loss = self.criterion(mean, y, sd**2)
                    self.val_loss.update(loss.item(), x.size(0))

            if self.val_loss.avg < best_loss:
                best_loss = self.val_loss.avg
                torch.save(self.model.state_dict(), 'best_model.pth')

            print(f'Epoch {epoch+1}/{iters} - Train Loss: {self.train_loss.avg:.4f}, Val Loss: {self.val_loss.avg:.4f}')

    def test(self):
        test_ds = PandasDataset(self.X_test, self.y_test)
        test_dl = DataLoader(test_ds, batch_size=len(test_ds)) # self.training_config['batchsize']?

        self.model.load_state_dict(torch.load('best_model.pth'))
        self.model.eval()

        ## Testing
        test_loss = AverageMeter()
        with torch.no_grad():
            for i, (x, y) in enumerate(test_dl):
                x, y = x.to(self.device), y.to(self.device)
                mean, sd = self.model(x)
                loss = self.criterion(mean, y, sd**2)
                test_loss.update(loss.item(), x.size(0))
        
        print(f'Test Loss: {test_loss.avg:.4f}')
    
    def predict(self, X, return_tensor=False):
        self.model.load_state_dict(torch.load('best_model.pth'))
        self.model.eval()

        pred_ds = PandasDataset(X)
        pred_dl = DataLoader(pred_ds, batch_size=len(pred_ds)) # self.training_config['batchsize']?

        with torch.no_grad():
            for i, x in enumerate(pred_dl):
                x = x.to(self.device)
                mean, sd = self.model(x)
            
            if return_tensor:
                mean_pred = mean
                sd_pred = sd
            else:
                mean_pred = mean.detach().cpu().numpy()
                sd_pred = sd.detach().cpu().numpy()
        
        return mean_pred, sd_pred