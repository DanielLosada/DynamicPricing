import torch
import os

import torch.optim as optim
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats

from utils import PandasDataset, AverageMeter, constraint_extrapolation_MLP
from distModel import distModel
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from torch.utils.data import DataLoader

# Define trainer class
class Trainer():
    torch.manual_seed(1)
    def __init__(self, X, y, model_config, training_config, n_preds=1, device='cpu'):
        self.device = device
        self.X = X
        self.y = y
        self.n_preds = n_preds

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
        
        # scale the data
        scaler = StandardScaler()

        self.X_train = pd.DataFrame(scaler.fit_transform(self.X_train), columns=scaler.feature_names_in_)
        self.X_mean = scaler.mean_
        self.X_std = scaler.scale_

        self.X_val = pd.DataFrame(scaler.transform(self.X_val), columns=scaler.feature_names_in_)
        if create_test:
            self.X_test = pd.DataFrame(scaler.transform(self.X_test), columns=scaler.feature_names_in_)
        
        self.y_train = pd.Series(scaler.fit_transform(self.y_train.values.reshape(-1, 1)).ravel(), name=self.y.name)
        self.y_mean = scaler.mean_
        self.y_std = scaler.scale_

        self.y_val = pd.Series(scaler.transform(self.y_val.values.reshape(-1, 1)).ravel(), name=self.y.name)
        if create_test:
            self.y_test = pd.Series(scaler.transform(self.y_test.values.reshape(-1, 1)).ravel(), name=self.y.name)

    
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
        patience = self.training_config.get('patience', 10)
        no_improvement = 0
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
                print(f'Validation loss improved from {best_loss:.4f} to {self.val_loss.avg:.4f} in epoch {epoch+1}. Saving model...')
                no_improvement = 0
                best_loss = self.val_loss.avg
                #self.best_model = self.model.state_dict() # copy.deepcopy(self.model.state_dict())
                torch.save(self.model.state_dict(), 'best_model.pth')
            else:
                no_improvement += 1
                if no_improvement >= patience:
                    print(f"Early stopping triggered after {epoch+1} epochs.")
                    break
            if epoch % 10 == 0:
                print(f'Epoch {epoch+1}/{iters} - Train Loss: {self.train_loss.avg:.4f}, Val Loss: {self.val_loss.avg:.4f}')
        self.model.load_state_dict(torch.load('best_model.pth'))

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
    
    def constraint_build(self):
        self.model.load_state_dict(torch.load('best_model.pth'))
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
        