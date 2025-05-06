import torch.nn as nn

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