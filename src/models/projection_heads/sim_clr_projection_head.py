import torch.nn as nn

class SimCLRProjectionHead(nn.Module):
    def __init__(self, layer_sizes=(2048, 2048, 128)):
        super(SimCLRProjectionHead, self).__init__()
        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            if i < len(layer_sizes) - 2:
                layers.append(nn.ReLU())
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)
