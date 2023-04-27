import pulp as pl
import dgl
import torch
import numpy as np

# 产地、销地的数量，以及产量和需求量
n_sources = 3
n_targets = 4
source_capacity = [500, 600, 250]
target_demand = [600, 400, 200, 150]

# 运费数据
costs = np.array([
    [20, 10, 12, 16],
    [22, 9, 9, 18],
    [23, 13, 10, 25]
])

# 创建问题
prob = pl.LpProblem("Transportation Problem", pl.LpMinimize)

# 定义变量
x = pl.LpVariable.dicts("x", [(i, j) for i in range(n_sources) for j in range(n_targets)], lowBound=0, cat='Continuous')

# 定义目标函数
prob += pl.lpSum(costs[i][j] * x[(i, j)] for i in range(n_sources) for j in range(n_targets))

# 定义约束条件
for i in range(n_sources):
    prob += pl.lpSum(x[(i, j)] for j in range(n_targets)) <= source_capacity[i]
for j in range(n_targets):
    prob += pl.lpSum(x[(i, j)] for i in range(n_sources)) == target_demand[j]

# 求解问题
prob.solve()

# 遍历变量，输出最优解
for i in range(n_sources):
    for j in range(n_targets):
        print(f"x[{i},{j}] = {x[(i, j)].varValue}")
print(f"Minimum Cost: {pl.value(prob.objective)}")

# 构建图神经网络模型
# 创建有向图
graph = dgl.DGLGraph()
graph.add_nodes(n_sources + n_targets)

# 添加产地到销地的有向边
for i in range(n_sources):
    for j in range(n_targets):
        edge_id = graph.number_of_edges()
        graph.add_edges(i, n_sources+j)
        graph.edges[edge_id].data['x'] = torch.tensor([x[(i, j)].varValue])

# 添加产地到汇点的有向边
for i in range(n_sources):
    edge_id = graph.number_of_edges()
    graph.add_edges(i, n_sources+n_targets)
    graph.edges[edge_id].data['x'] = torch.tensor([source_capacity[i]])

# 添加源点到销地的有向边
for j in range(n_targets):
    edge_id = graph.number_of_edges()
    graph.add_edges(n_sources+j, n_sources+n_targets+1)
    graph.edges[edge_id].data['x'] = torch.tensor([target_demand[j]])

# 定义图神经网络模型
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.lin1 = torch.nn.Linear(1, 8)
        self.lin2 = torch.nn.Linear(8, 1)

    def forward(self, x):
        x = self.lin1(x)
        x = torch.relu(x)
        x = self.lin2(x)
        return x

# 定义模型及优化器
net = Net()
optimizer = torch.optim.Adam(net.parameters(), lr=0.01)

# 训练模型
epochs = 1000
for epoch in range(epochs):
    # 前向传播
    y = net.forward(graph.edges['x'].unsqueeze(1).float())

    # 计算损失
    loss = torch.nn.functional.mse_loss(y.squeeze(1), torch.zeros_like(y.squeeze(1)))

    # 反向传播
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# 输出最小运费和对应的x值
y = net.forward(graph.edges['x'].unsqueeze(1).float())
min_cost = y[-1][0].item()
x_values = np.array([x[(i, j)].varValue for i in range(n_sources) for j in range(n_targets)])
print(f"Minimum Cost: {min_cost}")
print(f"x values: {x_values}")