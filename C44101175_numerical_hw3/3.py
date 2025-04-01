import numpy as np
import pandas as pd
from scipy.interpolate import KroghInterpolator

# 所有資料點
T_full = np.array([0, 3, 5, 8, 13])
D_full = np.array([0, 200, 375, 620, 990])
V_full = np.array([75, 77, 80, 74, 72])
speed_limit = 55 * 5280 / 3600  # ≈80.67 ft/s

# 建立資料點範圍
point_sets = {
    "使用全部5點": (T_full, D_full, V_full),
    "使用後4點": (T_full[1:], D_full[1:], V_full[1:]),
    "使用後3點": (T_full[2:], D_full[2:], V_full[2:])
}

# 儲存結果
results = {}

for label, (T_set, D_set, V_set) in point_sets.items():
    T_hermite, D_hermite = [], []
    for t, d, v in zip(T_set, D_set, V_set):
        T_hermite.extend([t, t])
        D_hermite.extend([d, v])
    
    # Hermite 插值器
    hermite_interp = KroghInterpolator(T_hermite, D_hermite)
    
    # (a) 預測 t = 10
    pos_10 = hermite_interp(10)
    vel_10 = hermite_interp.derivative(10)
    
    # (b) 找第一次超速時間與 (c) 最大速度
    t_vals = np.linspace(T_set[0], T_set[-1], 1000)
    v_vals = hermite_interp.derivative(t_vals)
    exceed_indices = np.where(v_vals > speed_limit)[0]
    first_exceed = t_vals[exceed_indices[0]] if exceed_indices.size > 0 else None
    max_vel = np.max(v_vals)
    
    results[label] = {
        "位置@t=10 (ft)": float(pos_10),
        "速度@t=10 (ft/s)": float(vel_10),
        "第一次超速時間 (s)": float(first_exceed) if first_exceed is not None else "未超速",
        "最大速度 (ft/s)": float(max_vel)
    }

# 用 pandas 印出成表格
df_results = pd.DataFrame(results).T
print(df_results)
