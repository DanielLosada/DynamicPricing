import torch
import os

import torch.optim as optim
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np

from utils import PandasDataset, AverageMeter, constraint_extrapolation_MLP
from distModel import distModel
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from torch.utils.data import DataLoader

# Define trainer class
class Trainer():
    torch.manual_seed(1)
    def __init__(self, X, y, config, device='cpu', path_model="./models/best_model.pth", id_col=None):
        X = X.reindex(sorted(X.columns), axis=1)

        self.device = device
        self.X = X
        self.y = y
        self.path_model = path_model

        self.config = config

        # Define the model
        if id_col is not None:
            self.id_col = id_col
            self.model = distModel(input_size=self.X.shape[1] -1, hidden_layers=self.config['hidden_layers'], hidden_size=self.config['hidden_size'], drop=self.config['drop']).to(self.device)
        else:
            self.model = distModel(input_size=self.X.shape[1], hidden_layers=self.config['hidden_layers'], hidden_size=self.config['hidden_size'], drop=self.config['drop']).to(self.device)
        # Define criterion: https://pytorch.org/docs/stable/generated/torch.nn.GaussianNLLLoss.html
        self.criterion = nn.GaussianNLLLoss()

    # Scaling adapted from DistCL, distcl.py: distcl.__init__.
    # Modification: separate X/y scalers and configurable sklearn partitions.
    # Source: https://github.com/antonioalcantaramata/DistCL/tree/main
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
        
        self.train_indices = X_train.index
        self.val_indices = X_val.index
        if create_test:
            self.test_indices = X_test.index
        
        # scale the data
        self.scaler_X = StandardScaler()

        self.X_train = pd.DataFrame(self.scaler_X.fit_transform(self.X_train), columns=self.scaler_X.feature_names_in_)
        self.X_mean = self.scaler_X.mean_
        self.X_std = self.scaler_X.scale_

        self.X_val = pd.DataFrame(self.scaler_X.transform(self.X_val), columns=self.scaler_X.feature_names_in_)
        if create_test:
            self.X_test = pd.DataFrame(self.scaler_X.transform(self.X_test), columns=self.scaler_X.feature_names_in_)
        
        self.scaler_y = StandardScaler()
        self.y_train = pd.Series(self.scaler_y.fit_transform(self.y_train.values.reshape(-1, 1)).ravel(), name=self.y.name)
        self.y_mean = self.scaler_y.mean_
        self.y_std = self.scaler_y.scale_

        self.y_val = pd.Series(self.scaler_y.transform(self.y_val.values.reshape(-1, 1)).ravel(), name=self.y.name)
        if create_test:
            self.y_test = pd.Series(self.scaler_y.transform(self.y_test.values.reshape(-1, 1)).ravel(), name=self.y.name)

    # Scaling adapted from DistCL, distcl.py: distcl.__init__.
    # Modification: split by ID and maintain separate X/y scalers.
    # Source: https://github.com/antonioalcantaramata/DistCL/tree/main
    def create_data_partitions_by_id(self, create_test=True, test_size=0.2, val_size=0.1, random_state=42):
        # Step 1: Get unique IDs
        unique_ids = self.X[self.id_col].unique()

        # Step 2: Split IDs
        train_ids, temp_ids = train_test_split(unique_ids, test_size=(test_size + val_size), random_state=random_state)
        val_ids, test_ids = train_test_split(temp_ids, test_size=test_size / (test_size + val_size), random_state=random_state)

        self.train_ids = np.sort(train_ids)
        self.val_ids = np.sort(val_ids)
        self.test_ids = np.sort(test_ids)

        # Step 3: Subset dataframe based on IDs
        self.train_indices = self.X[self.X['id'].isin(self.train_ids)].index
        self.val_indices = self.X[self.X['id'].isin(self.val_ids)].index
        self.test_indices = self.X[self.X['id'].isin(self.test_ids)].index

        self.X_train = self.X.loc[self.train_indices].drop(columns=[self.id_col])
        self.y_train = self.y.loc[self.train_indices]

        self.X_val = self.X.loc[self.val_indices].drop(columns=[self.id_col])
        self.y_val = self.y.loc[self.val_indices]

        if create_test:
            self.X_test = self.X.loc[self.test_indices].drop(columns=[self.id_col])
            self.y_test = self.y.loc[self.test_indices]
        
        # Step 4: Scale the data
        self.scaler_X = StandardScaler()
        self.X_train = pd.DataFrame(self.scaler_X.fit_transform(self.X_train), columns=self.scaler_X.feature_names_in_)
        self.X_mean = self.scaler_X.mean_
        self.X_std = self.scaler_X.scale_

        self.X_val = pd.DataFrame(self.scaler_X.transform(self.X_val), columns=self.scaler_X.feature_names_in_)
        if create_test:
            self.X_test = pd.DataFrame(self.scaler_X.transform(self.X_test), columns=self.scaler_X.feature_names_in_)

        self.scaler_y = StandardScaler()
        self.y_train = pd.Series(self.scaler_y.fit_transform(self.y_train.values.reshape(-1, 1)).ravel(), name=self.y.name)
        self.y_mean = self.scaler_y.mean_
        self.y_std = self.scaler_y.scale_

        self.y_val = pd.Series(self.scaler_y.transform(self.y_val.values.reshape(-1, 1)).ravel(), name=self.y.name)
        if create_test:
            self.y_test = pd.Series(self.scaler_y.transform(self.y_test.values.reshape(-1, 1)).ravel(), name=self.y.name)
    
    # Scaling adapted from DistCL, distcl.py: distcl.__init__.
    # Modification: split chronologically within each ID and use separate scalers.
    # Source: https://github.com/antonioalcantaramata/DistCL/tree/main
    def create_temporal_partitions_by_id(self, create_test=True, test_size=0.2, val_size=0.1):
        train_indices = []
        val_indices = []
        test_indices = []

        # Loop through each unique ID
        for uid in self.X[self.id_col].unique():
            df_id = self.X[self.X[self.id_col] == uid]
            idx = df_id.index.tolist()

            # Preserve the existing observation order within each client.

            n = len(idx)
            n_test = int(n * test_size)
            n_val = int(n * val_size)
            n_train = n - n_val - n_test

            train_indices.extend(idx[:n_train])
            val_indices.extend(idx[n_train:n_train + n_val])
            test_indices.extend(idx[n_train + n_val:])

        self.train_indices = train_indices
        self.val_indices = val_indices
        self.test_indices = test_indices

        # Drop ID column for training
        self.X_train = self.X.loc[self.train_indices].drop(columns=[self.id_col])
        self.y_train = self.y.loc[self.train_indices]
        print("self.X_train: ", self.X_train)

        self.X_val = self.X.loc[self.val_indices].drop(columns=[self.id_col])
        self.y_val = self.y.loc[self.val_indices]

        if create_test:
            self.X_test = self.X.loc[self.test_indices].drop(columns=[self.id_col])
            self.y_test = self.y.loc[self.test_indices]

        # Scale X
        self.scaler_X = StandardScaler()
        self.X_train = pd.DataFrame(self.scaler_X.fit_transform(self.X_train), columns=self.scaler_X.feature_names_in_)
        self.X_mean = self.scaler_X.mean_
        self.X_std = self.scaler_X.scale_

        self.X_val = pd.DataFrame(self.scaler_X.transform(self.X_val), columns=self.scaler_X.feature_names_in_)
        if create_test:
            self.X_test = pd.DataFrame(self.scaler_X.transform(self.X_test), columns=self.scaler_X.feature_names_in_)

        # Scale y
        self.scaler_y = StandardScaler()
        self.y_train = pd.Series(self.scaler_y.fit_transform(self.y_train.values.reshape(-1, 1)).ravel(), name=self.y.name)
        self.y_mean = self.scaler_y.mean_
        self.y_std = self.scaler_y.scale_

        self.y_val = pd.Series(self.scaler_y.transform(self.y_val.values.reshape(-1, 1)).ravel(), name=self.y.name)
        if create_test:
            self.y_test = pd.Series(self.scaler_y.transform(self.y_test.values.reshape(-1, 1)).ravel(), name=self.y.name)

    # Scaling adapted from DistCL, distcl.py: distcl.__init__.
    # Modification: accept explicit partitions and optional test data.
    # Source: https://github.com/antonioalcantaramata/DistCL/tree/main
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

        scaler = StandardScaler()
        self.X_train = pd.DataFrame(scaler.fit_transform(self.X_train), columns = scaler.feature_names_in_)
        
        self.X_mean = scaler.mean_
        self.X_std = scaler.scale_
        
        self.X_val = pd.DataFrame(scaler.transform(self.X_val), columns = scaler.feature_names_in_)
        if X_test is not None:
            self.X_test = pd.DataFrame(scaler.transform(self.X_test), columns = scaler.feature_names_in_)
        
        self.y_train = pd.DataFrame(scaler.fit_transform(self.y_train.values.reshape(-1, 1)), columns = [self.y.name])
        self.y_mean = scaler.mean_
        self.y_std = scaler.scale_
        
        self.y_val = pd.DataFrame(scaler.transform(self.y_val.values.reshape(-1, 1)), columns = [self.y.name])
        if y_test is not None:
            self.y_test = pd.DataFrame(scaler.transform(self.y_test.values.reshape(-1, 1)), columns = [self.y.name])

    # Adapted from DistCL, distnn.py: qdnn.train (training/validation loop).
    # Modifications: configurable training, PandasDataset, averaged losses,
    # early stopping, and a configurable checkpoint path.
    # Source: https://github.com/antonioalcantaramata/DistCL/tree/main
    def train(self):
        torch.cuda.empty_cache()

        train_ds = PandasDataset(self.X_train, self.y_train)
        val_ds = PandasDataset(self.X_val, self.y_val)

        train_dl = DataLoader(train_ds, batch_size = self.config['batchsize'], shuffle=True)
        val_dl = DataLoader(val_ds, batch_size=len(val_ds))

        iters = self.config['iters']
        patience = self.config.get('patience', 10)
        no_improvement = 0
        # Define optimizer: https://pytorch.org/docs/stable/generated/torch.optim.Adam.html
        optimizer = optim.Adam(self.model.parameters(), lr = self.config['lr'], weight_decay=self.config['wd'])

        best_loss = float('inf')
        for epoch in range(iters):
            ## Training
            self.model.train()
            self.train_loss = AverageMeter()
            for i, (x, y) in enumerate(train_dl):
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                mean, sd = self.model(x)
                loss = self.criterion(mean, y, sd**2)
                loss.backward()
                optimizer.step()
                self.train_loss.update(loss.item(), x.size(0))

            ## Validation
            self.model.eval()
            val_loss = AverageMeter()
            with torch.no_grad():
                for i, (x, y) in enumerate(val_dl):
                    x, y = x.to(self.device), y.to(self.device)
                    mean, sd = self.model(x)
                    loss = self.criterion(mean, y, sd**2)
                    val_loss.update(loss.item(), x.size(0))

            if val_loss.avg < best_loss:
                print(f'Validation loss improved from {best_loss:.4f} to {val_loss.avg:.4f} in epoch {epoch+1}. Saving model...')
                no_improvement = 0
                best_loss = val_loss.avg
                torch.save(self.model.state_dict(), self.path_model)
            else:
                no_improvement += 1
                if no_improvement >= patience:
                    print(f"Early stopping triggered after {epoch+1} epochs.")
                    break
            if epoch % 10 == 0:
                print(f'Epoch {epoch+1}/{iters} - Train Loss: {self.train_loss.avg:.4f}, Val Loss: {val_loss.avg:.4f}')
        self.model.load_state_dict(torch.load(self.path_model))

        return best_loss

    # Adapted from the test-evaluation block in DistCL, distnn.py: qdnn.train.
    # Modification: a separate method that reports averaged Gaussian NLL.
    # Source: https://github.com/antonioalcantaramata/DistCL/tree/main
    def test(self):
        test_ds = PandasDataset(self.X_test, self.y_test)
        test_dl = DataLoader(test_ds, batch_size=len(test_ds))

        self.model.load_state_dict(torch.load(self.path_model))
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
    
    # Adapted from the inference block in DistCL, distnn.py: qdnn.train.
    # Modifications: separate inference method, optional input scaling,
    # tensor output, and conversion back to the original target scale.
    # Source: https://github.com/antonioalcantaramata/DistCL/tree/main
    def predict(self, X, return_tensor=False, unscale=False, scale_data=False):
        self.model.load_state_dict(torch.load(self.path_model))
        self.model.eval()

        if scale_data:
            print("Scaling data...")
            X = pd.DataFrame(self.scaler_X.transform(X), columns=self.scaler_X.feature_names_in_)
        
        pred_ds = PandasDataset(X)
        pred_dl = DataLoader(pred_ds, batch_size=len(pred_ds))

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
            
            if unscale:
                mean_pred = mean_pred * self.y_std + self.y_mean
                sd_pred = sd_pred * self.y_std
        
        return mean_pred, sd_pred
    
    # Adapted from DistCL, distcl.py: distcl.constraint_build.
    # Modification: load the checkpoint and use self.model instead of a model argument.
    # Source: https://github.com/antonioalcantaramata/DistCL/tree/main
    def constraint_build(self):
        self.model.load_state_dict(torch.load(self.path_model))
        weight_names = []
        weight_values = []
        bias_names = []
        bias_values = []
        for name, param in self.model.named_parameters():
            if 'weight' in name:
                weight_names.append(name)
                weight_values.append(param.cpu().detach().numpy())
            else:
                bias_names.append(name)
                bias_values.append(param.cpu().detach().numpy())
        
        constraints = constraint_extrapolation_MLP(weight_values,bias_values,weight_names)
        
        return constraints
    
    def evaluate_gaussian_nll(self):
        """
        Evaluates the Gaussian NLL loss for a model and a baseline that assumes all outputs
        come from a single Gaussian with mean and std estimated from the training set.

        Args:
            
            device: Torch device ('cpu' or 'cuda') to perform evaluation on.

        Returns:
            A tuple (baseline_loss_value, model_loss_value)
        """
        # Convert training targets to tensor
        y_train_tensor = torch.tensor(self.y_train.values, dtype=torch.float32).unsqueeze(1).to(self.device)
        
        # Compute baseline mean and std from training targets
        baseline_mean = torch.mean(y_train_tensor)
        baseline_std = torch.std(y_train_tensor) + 1e-6  # add small epsilon to avoid zero variance

        # Define loss criterion
        criterion = nn.GaussianNLLLoss()

        # Evaluate baseline loss on validation targets
        y_val_tensor = torch.tensor(self.y_val.values, dtype=torch.float32).unsqueeze(1).to(self.device)
        baseline_mean_val = baseline_mean.repeat(y_val_tensor.shape[0], 1)
        baseline_var_val = (baseline_std ** 2).repeat(y_val_tensor.shape[0], 1)
        baseline_loss = criterion(baseline_mean_val, y_val_tensor, baseline_var_val)

        # Predict mean and std using the model
        pred_mean, pred_sd = self.predict(self.X_val, return_tensor=True)
        pred_mean, pred_sd = pred_mean.to(self.device), pred_sd.to(self.device)
        pred_var = pred_sd ** 2

        # Evaluate model loss
        model_loss = criterion(pred_mean, y_val_tensor, pred_var)

        print(f"Baseline Gaussian NLL Loss: {baseline_loss.item():.4f}")
        print(f"Model Gaussian NLL Loss: {model_loss.item():.4f}")

        return baseline_loss.item(), model_loss.item()
    
    def evaluate_standardized_residuals(self, output_dir="residual_diagnostics", prefix=""):
        """
        Evaluates standardized residuals by plotting histogram, QQ-plot,
        and performing a KS test against the standard normal distribution.

        Args:
            output_dir (str): Directory to save images.
            prefix (str): Optional prefix for filenames

        Returns:
            ks_stat (float): KS test statistic
            p_value (float): KS test p-value
        """
        y_val_tensor = torch.tensor(self.y_val.values, dtype=torch.float32).unsqueeze(1).to(self.device)
        pred_mean, pred_sd = self.predict(self.X_val, return_tensor=True)
        pred_mean, pred_sd = pred_mean.to(self.device), pred_sd.to(self.device)
        pred_var = pred_sd ** 2

        os.makedirs(output_dir, exist_ok=True)

        # Compute standardized residuals
        z = (y_val_tensor - pred_mean) / pred_sd
        z_np = z.squeeze().detach().cpu().numpy()

        # === Histogram ===
        plt.figure()
        plt.hist(z_np, bins=30, density=True, alpha=0.6, label='Standardized Residuals (Z-scores)')
        
        x = torch.linspace(-5, 5, 1000).numpy()
        plt.plot(x, stats.norm.pdf(x), 'k--', label='Standard Normal')
        plt.title("Histogram of Standardized Residuals")
        plt.xlabel("Z-score")
        plt.ylabel("Density")
        plt.legend()
        plt.grid(True)
        hist_path = os.path.join(output_dir, f"{prefix}z_histogram.png")
        plt.savefig(hist_path)
        plt.close()

        # === QQ-Plot ===
        plt.figure()
        stats.probplot(z_np, dist="norm", plot=plt)
        plt.title("QQ-Plot of Standardized Residuals")
        qq_path = os.path.join(output_dir, f"{prefix}z_qqplot.png")
        plt.savefig(qq_path)
        plt.close()

        # === KS Test ===
        ks_stat, p_value = stats.kstest(z_np, 'norm')

        print(f"KS test against N(0,1): statistic = {ks_stat:.4f}, p-value = {p_value:.4f}")
        if p_value > 0.05:
            print("✅ The standardized residuals do not significantly differ from a standard normal distribution — model variance estimates appear well-calibrated.")
        else:
            print("⚠️ The standardized residuals significantly differ from a standard normal distribution — model variance estimates may be miscalibrated.")

        print(f"Saved: {hist_path}")
        print(f"Saved: {qq_path}")

        return ks_stat, p_value
        