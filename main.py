import numpy as np

from model import MSEModel, CosineModel, NeuralRegressionModel, AffineProductModel

import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
from matplotlib import pyplot as plt

data_num_vertices = "mix"
basis_num_vertices = "mix"
encoder_type = "rotscale"
basis_size = 32
train_size = 10000
test_size = 1000

print("Experiment encoder type {}, basis number {}, basis type {}".format(encoder_type, basis_num_vertices, basis_size))

##### Training #####
batch_size = 512
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("Data Preparation:")
data = np.load("dataset/de_{}-be_{}-et_{}-bs_{}-trs_{}-tes_{}.npz".format(data_num_vertices, basis_num_vertices, encoder_type, basis_size, train_size, test_size))
# data = np.load("dataset/{}-{}-{}-{}.npz".format(encoder_type, basis_size, train_size, test_size))
# data = np.load("dataset/{}-{}-{}.npz".format(encoder_type, train_size, test_size))

train_encodings = data["train_encodings"]
train_targets = data["train_targets"]

print(train_encodings.shape)

test_encodings = data["test_encodings"]
test_targets = data["test_targets"]

train_encodings, train_targets = torch.FloatTensor(train_encodings).to(device), torch.FloatTensor(train_targets).to(device)
test_encodings, test_targets = torch.FloatTensor(test_encodings).to(device), torch.FloatTensor(test_targets).to(device)

print("Model Initialization:")
model = CosineModel(train_encodings.shape[1]).to(device)

criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, weight_decay=1e-5)

print("Start Training:")
total_loss = 0
mses, hits, mrrs = [], [], []

for i in tqdm(range(1000000)):
    model.train()

    optimizer.zero_grad()  # zero the gradient buffers

    idx1, idx2 = np.random.choice(train_size, 512, replace=False), np.random.choice(train_size, 512, replace=False)
    train_encodings1, train_encodings2 = train_encodings[idx1], train_encodings[idx2]
    targets = train_targets[idx1, idx2]

    scores = model(train_encodings1, train_encodings2)
    loss = criterion(scores, targets)

    total_loss += loss.item()

    loss.backward()
    optimizer.step()

    if (i + 1) % 10000 == 0:
        print("\nTraining MSE: ", total_loss / 10000)
        total_loss = 0
        total_preds, total_targets = [], []

        with torch.no_grad():
            mse = 0.
            hit10 = 0.
            mrr = 0.
            model.eval()

            size = 1000
            for i in range(size):
                test_encodings1, test_encodings2 = test_encodings[i].reshape(1,-1) * torch.ones((size, train_encodings.shape[1])), test_encodings[:size]

                scores = model(test_encodings1, test_encodings2)
                pred_orders = torch.argsort(scores)
                target_orders = torch.argsort(test_targets[i, :size])

                mse += criterion(scores, test_targets[i, :size]).item()

                hit10 += len(set(target_orders[1:11].detach().numpy().tolist()) & set(pred_orders[1:101].detach().numpy().tolist())) / 10

                ranks = []
                for to in target_orders[1:101]:
                    ranks.append(np.where(pred_orders == to)[0])

                mrr += 1 / np.min(ranks)

                total_preds.append(scores.detach().numpy().tolist())
                total_targets.append(test_targets[i, :size].detach().numpy().tolist())

            mses.append(mse / size)
            hits.append(hit10 / size)
            mrrs.append(mrr / size)
            print("Test MSE: ", mse / size, "Test Hit@10: ", hit10 / size, "Test MRR: ", mrr / size)

        np.savez("eval/eval-de_{}-be_{}-et_{}-bs_{}".format(data_num_vertices, basis_num_vertices, encoder_type, batch_size), total_preds=total_preds, total_targets=total_targets)

print(f"Best MSE: {np.min(mses):.4f}, Best Hit@100: {np.max(hits):.4f} Best MRR: {np.max(mrrs):.4f}")