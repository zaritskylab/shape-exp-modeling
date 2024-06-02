import torch
import torch.nn as nn

class SimpleLinearNet(nn.Module):
    '''
    linear net mapping cell-type and batch to protein expression profile.
    Arguments:
        in_features : num fts to input
        out_features : num proteins tagged and quantified
    
    '''
    
    
    def __init__(self, in_features = 30, out_features = 48):
        super(SimpleLinearNet, self).__init__()
        #first layer
        self.Embed = nn.Linear(in_features, 180)
        self.bn1 = nn.BatchNorm1d(num_features=180)
        #second layer
        self.fc1 = nn.Linear(180, 90)
        self.bn2 = nn.BatchNorm1d(num_features=90)
        # third layer
        self.fc2 = nn.Linear(90, 90)
        self.bn3 = nn.BatchNorm1d(num_features=90)
        #activation
        self.relu = nn.ReLU()
        #last layer
        self.out = nn.Linear(90, out_features)
        
    def forward(self, x):
        
        x = self.relu(self.bn1(self.Embed(x)))
        x = self.relu(self.bn2(self.fc1(x)))
        x = self.relu(self.bn3(self.fc2(x)))
                      
        predictions = self.out(x)
        
        return predictions