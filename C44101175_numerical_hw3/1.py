import numpy as np
import pandas as pd
from scipy.interpolate import lagrange
from math import factorial

# 給定資料點
x_values = np.array([0.698, 0.733, 0.768, 0.803])
y_values = np.array([0.7661, 0.7432, 0.7193, 0.6946])

# 目標點
x_target = 0.750
true_value = 0.7317

# 建立不同階數的 Lagrange 多項式並計算誤差
results = []
for degree in range(1, 5):
    x_subset = x_values[:degree+1]
    y_subset = y_values[:degree+1]

    poly = lagrange(x_subset, y_subset)
    approx = poly(x_target)
    abs_error = abs(true_value - approx)

    # 計算 error bound（理論值）
    max_derivative = 1  # cos(x) 的導數不超過 1
    product_term = np.prod([x_target - xi for xi in x_subset])
    bound = abs(max_derivative / factorial(degree + 1) * product_term)

    results.append({
        "degree": degree,
        "approximation": approx,
        "abs_error": abs_error,
        "error_bound": bound
    })

# 顯示結果表格
df = pd.DataFrame(results)
print(df)
