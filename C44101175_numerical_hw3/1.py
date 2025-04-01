import numpy as np
import pandas as pd
from itertools import combinations
from scipy.interpolate import lagrange
from math import factorial

# 已知的 (x, cos(x)) 資料
data_points = {
    0.698: 0.7661,
    0.733: 0.7432,
    0.768: 0.7193,
    0.803: 0.6946
}

# 目標點與真實值
x_target = 0.750
true_value = 0.7317
M = 1  # cos(x) 所有導數最大值 ≤ 1

# 儲存所有結果
results = []

# 建立1~3階的Lagrange多項式與誤差界限
for deg in range(1, 4):  # 1 到 3 次
    combs = list(combinations(data_points.items(), deg + 1))
    for combo in combs:
        x_vals = np.array([pt[0] for pt in combo])
        y_vals = np.array([pt[1] for pt in combo])
        poly = lagrange(x_vals, y_vals)
        approx = poly(x_target)
        error = abs(approx - true_value)

        # 計算理論誤差界限
        product_term = np.prod([abs(x_target - xi) for xi in x_vals])
        bound = (M / factorial(deg + 1)) * product_term

        results.append({
            "Degree": deg,
            "Points Used": ", ".join([f"{x:.3f}" for x in x_vals]),
            "Approximation": round(approx, 6),
            "Absolute Error": round(error, 6),
            "Error Bound": round(bound, 6)
        })

# 用 pandas 表格輸出
df = pd.DataFrame(results)
df = df.sort_values(by=["Degree", "Absolute Error"])
print(df.to_string(index=False))

