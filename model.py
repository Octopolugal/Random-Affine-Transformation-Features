import torch
import torch.nn as nn

class MSEModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(144, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 8),
            nn.ReLU()
        )

        self.linear = nn.Linear(8, 1)
        # self.linear = nn.Sequential(
        #     nn.Linear(32, 16),
        #     nn.ReLU(),
        #     nn.Linear(16, 1)
        # )

    def forward(self, e1, e2):
        e1, e2 = self.projection(e1), self.projection(e2)

        return self.linear(torch.abs(e1 - e2)).squeeze()
        # return torch.sum((e1 @ self.M) * e2, dim=1)
        # return torch.mean(torch.abs(e1 - e2), dim=1)

class CosineModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 8)
        )

        self.sim_fun = nn.CosineSimilarity(dim=1, eps=1e-6)
        # self.dropout = nn.Dropout(0.3)

    def forward(self, e1, e2):
        e1, e2 = self.projection(e1), self.projection(e2)
        # return torch.sum((e1 @ self.M) * e2, dim=1)
        # return self.dropout(1 - self.sim_fun(e1, e2))
        return 1 - self.sim_fun(e1, e2)

class NeuralRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(144, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )

        self.regression = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )

    def forward(self, e1, e2):
        e1, e2 = self.projection(e1), self.projection(e2)
        return self.regression(e1 - e2).squeeze()

class AffineProductModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(144, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )

        w = torch.empty(32, 32)
        nn.init.normal_(w)
        self.M = nn.Parameter(w)

    def forward(self, e1, e2):
        e1, e2 = self.projection(e1), self.projection(e2)
        # return torch.sum((e1 @ self.M) * e2, dim=1)
        return torch.mean(torch.abs((e1 @ self.M) - e2), dim=1)