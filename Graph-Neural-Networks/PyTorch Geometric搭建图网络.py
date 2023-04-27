import numpy as np
from pulp import LpVariable, LpProblem, LpMinimize, lpSum
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, GATConv

# 运输问题代码
m = 3
n = 4
a = np.array([500, 600, 250])
b = np.array([600, 400, 200, 150])
c = np.array([
    [20, 10, 12, 16],
    [22, 9, 9, 18],
    [23, 13, 10, 25]
])

# 定义变量
x = np.array([[LpVariable(f"x{i}{j}", lowBound=0) for j in range(n)] for i in range(m)])

# 定义问题
prob = LpProblem("Transportation Problem", LpMinimize)

# 定义目标函数
prob += lpSum(c[i][j] * x[i][j] for i in range(m) for j in range(n))

# 定义供应量与需求量的约束条件
for i in range(m):
    prob += lpSum(x[i][j] for j in range(n)) == a[i]
for j in range(n):
    prob += lpSum(x[i][j] for i in range(m)) == b[j]

prob.solve()

print(f"Optimal Value: {prob.objective.value()}")

for i in range(m):
    for j in range(n):
        print(f"x{i}{j} = {x[i][j].value()}")

    # 构建图
edge_index = np.array([
    [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2],
    [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3]
])
edge_weight = np.array([x[i][j].varValue for i in range(m) for j in range(n)])
edge_attr = torch.tensor(edge_weight, dtype=torch.float)

x_dim = m + n
x = torch.zeros((x_dim,), dtype=torch.float)
x[:m] = torch.tensor(a, dtype=torch.float)
x[m:] = torch.tensor(-b, dtype=torch.float)

data = Data(x=x, edge_index=torch.tensor(edge_index, dtype=torch.long), edge_attr=edge_attr)


# GCN模型
class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index, edge_attr):
        x = self.conv1(x, edge_index, edge_weight=edge_attr)
        x = F.relu(x)
        x = self.conv2(x, edge_index, edge_weight=edge_attr)
        return x

    # GAT模型


class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GAT, self).__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=4)
        self.conv2 = GATConv(hidden_channels, out_channels, heads=2)

    def forward(self, x, edge_index, edge_attr):
        x = self.conv1(x, edge_index, edge_weight=edge_attr)
        x = F.elu(x)
        x = self.conv2(x, edge_index, edge_weight=edge_attr)
        return x

    # 运行模型


device = torch.device('cpu')

# GCN模型
model_gcn = GCN(in_channels=x_dim, hidden_channels=64, out_channels=1).to(device)
optimizer_gcn = torch.optim.Adam(model_gcn.parameters(), lr=0.01)

# GAT模型
model_gat = GAT(in_channels=x_dim, hidden_channels=64, out_channels=1).to(device)
optimizer_gat = torch.optim.Adam(model_gat.parameters(), lr=0.01)

# 训练模型
model_gcn.train()
model_gat.train()

for epoch in range(200):
    optimizer_gcn.zero_grad()
    out_gcn = model_gcn(data.x.to(device), data.edge_index.to(device), data.edge_attr.to(device))
    loss_gcn = -F.mse_loss(out_gcn, torch.zeros(out_gcn.size(), dtype=torch.float))
    loss_gcn.backward()
    optimizer_gcn.step()

    optimizer_gat.zero_grad()
    out_gat = model_gat(data.x.to(device), data.edge_index.to(device), data.edge_attr.to(device))
    loss_gat = -F.mse_loss(out_gat, torch.zeros(out_gat.size(), dtype=torch.float))
    loss_gat.backward()
    optimizer_gat.step()

# 打印结果
print("GCN model:")
print(f"Optimal Value: {-out_gcn.item()}")
for i in range(m):
    for j in range(n):
        print(f"x{i}{j} = {out_gcn[i * n + j].item()}")

print("\nGAT model:")
print(f"Optimal Value: {-out_gat.item()}")
for i in range(m):
    for j in range(n):
        print(f"x{i}{j} = {out_gat[i * n + j].item()}")