import numpy as np

def transportation_solver(supply, demand, costs):
    # 构造初始解
    m, n = len(supply), len(demand)
    nm = m + n
    S, D = n, m + n
    A = np.zeros((nm, nm), dtype=int)
    B = np.zeros(nm, dtype=int)
    for i in range(nm):
        if i < S:
            B[i] = supply[i]
            A[i][i] = 1
            A[i][nm + i] = 1
        elif D <= i < D + S:
            B[i] = -demand[i - D]
            A[i][i] = -1
            A[i][nm + i] = 1
        else:
            A[i][i] = -1
            A[i][i - S] = 1
            A[i][nm + i - D] = 1

    # 计算初始解的总成本
    c = np.reshape(costs, (nm, nm))
    x0 = np.zeros((nm, nm), dtype=int)
    for i in range(nm):
        for j in range(nm):
            if A[i][j] == 1:
                x0[i][j] = min(B[i], B[j + nm])
                B[i] -= x0[i][j]
                B[j + nm] -= x0[i][j]
    z0 = np.sum(c * x0)

    # 输出初始解的信息
    print("初始解：")
    print("总成本：", z0)
    for i in range(nm):
        for j in range(nm):
            if x0[i][j] != 0:
                print("    x[{}][{}] = {}".format(i, j, x0[i][j]))

    # 对初始解进行改进
    while True:
        # 计算对偶变量和单位费用矩阵
        u = np.zeros(nm, dtype=int)
        v = np.zeros(nm, dtype=int)
        for i in range(nm):
            for j in range(nm):
                if A[i][j] == 1:
                    if u[i] == 0 and v[j] == 0:
                        u[i] = c[i][j]
                    elif u[i] != 0 and v[j] == 0:
                        v[j] = c[i][j] - u[i]
                    elif u[i] == 0 and v[j] != 0:
                        u[i] = c[i][j] - v[j]
        rc = c - np.reshape(u, (nm, 1)) - np.reshape(v, (1, nm))
        min_rc = np.min(rc)
        if min_rc >= 0:
            # 原问题的最优解已经得到
            break

        # 构造改进路
        i, j = np.where(rc == min_rc)
        i, j = i[0], j[0]
        d = min(B[i], B[j + nm])
        if d == B[i]:
            p = i
        else:
            p = j + nm
        path = [p]
        while True:
            q = -1
            for k in range(nm):
                if A[p][k] == 1 and rc[p][k] < 0:
                    if q == -1:
                        q = k
                    elif rc[p][k] < rc[p][q]:
                        q = k
            if q == -1:
                break
            path.append(q)
            p = q
        w = np.zeros(nm, dtype=int)
        w[path[0]] = d
        for k in range(len(path)-1):
            if A[path[k]][path[k+1]] == 1:
                w[path[k+1]] = -w[path[k]]
        delta = min([B[i] if w[i] > 0 else -B[j+nm] for i in range(nm) for j in range(nm) if A[i][j] == 1 and w[i] * w[j+nm] < 0])
        x0 = x0 + delta * np.reshape(w, (nm, 1)) * np.reshape(w, (1, nm))
        B = B - delta * w
        z0 = np.sum(c * x0)

    # 输出最优解的信息
    print("最优解：")
    print("总成本：", z0)
    for i in range(nm):
        for j in range(nm):
            if x0[i][j] != 0:
                print("    x[{}][{}] = {}".format(i, j, x0[i][j]))