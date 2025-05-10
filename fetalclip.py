
import torch.nn as nn

class FetalCLIP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(224 * 224, 3)

    def forward(self, x):
        return self.layer(x.view(x.size(0), -1))
