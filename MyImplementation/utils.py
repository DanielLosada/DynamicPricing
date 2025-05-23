import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd

from torch.utils.data import Dataset
from pyomo.environ import *


np.random.seed(42)
torch.manual_seed(1)

def load_case_study_data():
    data = pd.read_csv('../DistCL_code/data_house_japan/data_clean.csv', index_col=0)
    data.index = pd.to_datetime(data.index)

    y = data['demand']
    X = data.drop(['demand'], axis=1, inplace=False)

    val_ind = X.index[(X.index >= pd.to_datetime('2018-11-01')) & (X.index < pd.to_datetime('2018-12-01'))]
    test_ind = X.index[X.index >= pd.to_datetime('2018-12-01')]

    return X, y, val_ind, test_ind

'''def generate_simple_synthetic_data(n_samples=1000, seed=42):
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

        return X, y, true_mean, true_std'''

def generate_simple_synthetic_data(n_samples=1000, seed=42):
    np.random.seed(seed)

    # Contextual features
    x1 = np.random.uniform(0, 1, size=n_samples)   # e.g., income level
    x2 = np.random.uniform(0, 1, size=n_samples)   # e.g., brand loyalty

    # Price feature (bounded between realistic min and max)
    price = np.random.uniform(10, 30, size=n_samples)

    # Demand decay using sigmoid centered at mean price 17
    k = 0.35
    sigmoid = 1 / (1 + np.exp(k * (price - 17)))
    valid_price_mask = (price >= 10) & (price <= 30)
    base_demand = (5 * x1 + 3 * x2) * sigmoid * 2

    true_mean = np.where(valid_price_mask, base_demand, 0)

    # Bell-shaped variability (highest near price 17)
    std_peak = 1.0
    std_min = 0.2
    width = 5

    base_std = std_min + (std_peak - std_min) * np.exp(-((price - 17)**2) / (2 * width**2))
    true_std = np.where(valid_price_mask, base_std, 0)

    # Sample demand, clip, round
    y = np.random.normal(loc=true_mean, scale=true_std)
    y = np.maximum(0, y)
    #y = np.round(y).astype(int)

    # Package
    X = pd.DataFrame({'x1': x1, 'x2': x2, 'price': price})
    y = pd.Series(y, name='y')

    return X, y, true_mean, true_std

def generate_data_by_ids(n_ids=100, n_prices_per_id=10, seed=42):
    np.random.seed(seed)

    # Create one (x1, x2) pair per ID
    x1_vals = np.random.uniform(0, 1, size=n_ids)
    x2_vals = np.random.uniform(0, 1, size=n_ids)
    ids = np.arange(1, n_ids + 1)

    # Create all combinations of ID and price
    data = []
    for i in range(n_ids):
        id_ = ids[i]
        x1 = x1_vals[i]
        x2 = x2_vals[i]

        # Generate n_prices_per_id random prices between 10 and 30
        prices = np.random.uniform(10, 30, size=n_prices_per_id)

        for price in prices:
            # Demand decay using sigmoid
            k = 0.35
            center = 17
            sigmoid = 1 / (1 + np.exp(k * (price - center)))

            # Base demand, scaled up
            raw_demand = (5 * x1 + 3 * x2)
            base_demand = raw_demand * sigmoid * 2  # scaling factor

            # Mean
            true_mean = base_demand if 10 <= price <= 30 else 0

            # Std: bell-shaped curve, low at extremes
            std_peak = 1.0
            std_min = 0.2
            width = 5
            base_std = std_min + (std_peak - std_min) * np.exp(-((price - center) ** 2) / (2 * width ** 2))
            true_std = base_std if 10 <= price <= 30 else 0

            # Sample demand
            y = np.random.normal(loc=true_mean, scale=true_std)
            y = max(0, y)

            data.append({
                'id': id_,
                'x1': x1,
                'x2': x2,
                'price': price,
                'true_mean': true_mean,
                'true_std': true_std,
                'y': y
            })

    df = pd.DataFrame(data)
    return df

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
            y = torch.tensor(self.y[idx], dtype=torch.float32).reshape(1)
            return x, y
        return x

class TabularDataset(Dataset):
    def __init__(self, X, y):
        """
        Characterizes a Dataset for PyTorch
        """
        
        self.n = X.shape[0]
        
        self.y = y.astype(np.float32).values.reshape(-1, 1)
        self.X = X.astype(np.float32).values
    
    def __len__(self):
        """
        Denotes the total number of samples.
        """
        return self.n
    
    def __getitem__(self, idx):
        """
        Generates one sample of data.
        """
        return [self.X[idx], self.y[idx]]



def const_embed(model, opt_model, constaints, outcome, n_scenarios = 1, deterministic = False):
        '''
        This function embdeds the fitted prediction model within the optimization problem.
        Expecting a defined optimization model "opt_model", constraint dataframe "constraints",
        a name for an "outcome", and the number of scenarios to generate.
        '''
        
        # Predefining variable y
        #opt_model.y = Var(Any, dense=False, domain=Reals)
        
        M_l=-1e3
        M_u=1e3
        opt_model.v = Var(Any, dense=False, domain=NonNegativeReals)
        opt_model.v_ind = Var(Any, dense=False, domain=Binary)
        
        for n in range(model.n_preds):
            max_layer = max(constaints['layer'])
            nodes_input = range(len(model.X_train.columns))
            v_input = [(opt_model.x[name,n]-model.X_mean[i])/model.X_std[i] for i,name in enumerate(model.X_train.columns)]
            
            for l in range(max_layer):
                df_layer = constaints.query('layer == %d' % l)
                max_nodes = [k for k in df_layer.columns if 'node_' in k]
                # coeffs_layer = np.array(df_layer.iloc[:, range(len(max_nodes))].dropna(axis=1))
                coeffs_layer = np.array(df_layer.loc[:, max_nodes].dropna(axis=1))
                intercepts_layer = np.array(df_layer['intercept'])
                nodes = df_layer['node']
                
                if l == max_layer-1:
                    node = nodes.iloc[0] 
                    opt_model.add_component('Mean_est' + str(n), Constraint(rule=opt_model.y[outcome, n,'mean'] == (sum(v_input[i] * coeffs_layer[node, i] for i in nodes_input) + intercepts_layer[node])))
                    
                    l = l+1
                    df_layer = constaints.query('layer == %d' % l)
                    max_nodes = [k for k in df_layer.columns if 'node_' in k]
                    coeffs_layer = np.array(df_layer.loc[:, max_nodes].dropna(axis=1))
                    intercepts_layer = np.array(df_layer['intercept'])
                    opt_model.add_component('Var_est' + str(n), Constraint(rule=opt_model.y[outcome, n,'sd'] == (sum(v_input[i] * coeffs_layer[node, i] for i in nodes_input) + intercepts_layer[node])))
                    
                else:
                    # Save v_pos for input to next layer
                    v_pos_list = []
                    for node in nodes:
                        ## Initialize variables
                        v_pos_list.append(opt_model.v[(outcome, l, node, n)])
                        opt_model.add_component('constraint_1_layer' + str(l) + '_node' + str(node)+ '_pred'+ str(n)+ outcome,
                                            Constraint(rule=opt_model.v[(outcome, l, node, n)] >= sum(v_input[i] * coeffs_layer[node, i] for i in nodes_input) + intercepts_layer[node]))
                        opt_model.add_component('constraint_2_layer' + str(l) + '_node' + str(node)+ '_pred'+ str(n)+ outcome,
                                            Constraint(rule=opt_model.v[(outcome, l, node, n)] <= M_u * (opt_model.v_ind[(outcome, l, node, n)])))
                        opt_model.add_component('constraint_3_layer' + str(l) + '_node' + str(node)+ '_pred'+ str(n)+ outcome,
                                            Constraint(rule=opt_model.v[(outcome, l, node, n)] <= sum(v_input[i] * coeffs_layer[node, i] for i in nodes_input) + intercepts_layer[node] - M_l * (1 - opt_model.v_ind[(outcome, l, node, n)])))
                    ## Prepare nodes_input for next layer
                    nodes_input = nodes
                    v_input = v_pos_list
        
        
        if deterministic == False:
            np.random.seed(0)
            z_vals = np.random.normal(size=n_scenarios)
            
            for n in range(model.n_preds):
                for w in range(1, n_scenarios + 1):
                    # opt_model.y[outcome, n,'sd'] and opt_model.y[outcome, n,'mean'] are already defined in the model as learned variables above. Samples from standard normal distribution are added to the mean and sd of the prediction.
                    # The mean and sd are scaled to the original scale of the data using self.y_std and self.y_mean.
                    opt_model.add_component(outcome + '_pred' + str(n) + '_sce' + str(w), Constraint(rule=opt_model.y[outcome, n, w] == (z_vals[w-1] * opt_model.y[outcome, n,'sd'] + opt_model.y[outcome, n,'mean'])*model.y_std + model.y_mean))
        else:
            for n in range(model.n_preds):
                for w in range(1, n_scenarios + 1):
                    opt_model.add_component(outcome + '_pred' + str(n) + '_sce' + str(w), Constraint(rule=opt_model.y[outcome, n, w] == opt_model.y[outcome, n,'mean']*model.y_std + model.y_mean))