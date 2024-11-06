import torch
from torch import nn

class LinearRegression(nn.Module):
    def __init__(self, num_features):
        super(LinearRegression, self).__init__()  
        self.num_features = num_features
        self.weights = torch.randn(num_features, 1)
        self.bias = torch.zeros(1)
        
        # Define the layers
        self.fc1 = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.Dropout(0.3),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(512, 1024),
            nn.Dropout(0.3),
            nn.ReLU()
        )
        self.fc3 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.Dropout(0.3),
            nn.ReLU()
        )
        self.fc4 = nn.Linear(512, 6)
        
        self.initialize_weights()  
    
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)  
                nn.init.zeros_(m.bias)  

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        #x = nn.Softmax(dim=1)(x) No Softmax for CrossEntropy 
        return x
    
