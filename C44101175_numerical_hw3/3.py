import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator

# 已知資料：時間（秒）、位置（英尺）
T = np.array([0, 3, 5, 8, 13])
D = np.array([0, 200, 375, 620, 990])

# 使用 PCHIP 為位置做插值（位置-時間曲線）
pchip_position = PchipInterpolator(T, D)

# 對位置函數做微分以取得速度函數
pchip_velocity = pchip_position.derivative()

# === (a) t = 10 秒時的位置與速度 ===
t_predict = 10
position_at_10 = pchip_position(t_predict)
velocity_at_10 = pchip_velocity(t_predict)

print("=== (a) t = 10 秒時的位置與速度 ===")
print(f"位置：{position_at_10:.2f} ft")
print(f"速度：{velocity_at_10:.2f} ft/s\n")

# === (b) 是否超速與第一次超速的時間 ===
# 速限轉換：55 mi/h = 80.67 ft/s
speed_limit = 55 * 5280 / 3600

# 建立細緻時間點來分析整體速度變化
t_vals = np.linspace(0, 13, 1000)
v_vals = pchip_velocity(t_vals)

# 找出第一次超速的時間
exceed_indices = np.where(v_vals > speed_limit)[0]
first_exceed_time = t_vals[exceed_indices[0]] if exceed_indices.size > 0 else None

print("=== (b) 是否超過速限 (55 mi/h ≈ 80.67 ft/s) ===")
if first_exceed_time:
    print(f"是的，第一次超速發生在 {first_exceed_time:.2f} 秒")
else:
    print("否，車輛從未超速")
print()

# === (c) 最大速度 ===
max_velocity = np.max(v_vals)
print("=== (c) 預測最大速度 ===")
print(f"最大速度為：{max_velocity:.2f} ft/s")
