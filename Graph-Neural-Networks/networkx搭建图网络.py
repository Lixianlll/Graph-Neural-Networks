import pulp as pl
import networkx as nx
import matplotlib.pyplot as plt

# 定义产地、市场、产量和需求
A = ['1', '2', '3']
B = ['1', '2', '3', '4']
a = {'1': 500, '2': 600, '3': 250}
b = {'1': 600, '2': 400, '3': 200, '4': 150}

# 定义运费
f = {'1': {'1': 20, '2': 10, '3': 12, '4': 16},
     '2': {'1': 22, '2': 9, '3': 9, '4': 18},
     '3': {'1': 23, '2': 13, '3': 10, '4': 25}}

# 定义线性规划模型
model = pl.LpProblem("Transportation", pl.LpMinimize)

# 定义决策变量
x = pl.LpVariable.dicts("Route", [(i, j) for i in A for j in B],
                        lowBound=0, cat='Continuous')

# 定义目标函数
model += pl.lpSum([x[i, j]*f[i][j] for i in A for j in B]), "Total Cost"

# 定义约束条件
for i in A:
   model += pl.lpSum([x[i, j] for j in B]) == a[i], "Sum of Products %s"%i

for j in B:
   model += pl.lpSum([x[i, j] for i in A]) == b[j], "Sum of Demand %s"%j

# 求解线性规划问题
model.solve()

# 打印结果
print("Total Cost = ", pl.value(model.objective))
for i in A:
    for j in B:
        print("路线 %s -> %s 量为 %d" % (i, j, x[i, j].varValue))

# 构建有向图
G = nx.DiGraph()

# 添加节点
for i in A:
    for j in B:
        G.add_node("%s -> %s" % (i, j))

# 添加边
for i in A:
    for j in B:
        for k in A:
            for l in B:
                if x[i, j].varValue > 0 and (i, j) != (k, l):
                    if i == k:
                        G.add_edge("%s -> %s" % (i, j), "%s -> %s" % (k, l),
                                   weight=f[i][j], label=str(int(x[i, j].varValue)))
                    elif j == l:
                        G.add_edge("%s -> %s" % (i, j), "%s -> %s" % (k, l),
                                   weight=f[i][j], label=str(int(x[i, j].varValue)))

# 绘制有向图
pos = nx.spring_layout(G, k=0.5)
labels = nx.get_edge_attributes(G, 'label')
nx.draw_networkx(G, pos)
nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
plt.show()
