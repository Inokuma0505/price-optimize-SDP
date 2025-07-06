import time
import numpy as np


from src.create_data.create_data import create_data
from src.linear_regression.build_model import build_model
from src.ex_price_optimize.ex_optimize_price import optimize_price
from src.new_price_optimize.new_optimize_price import optimize_price_new

alpha, beta, X, Y = create_data(M=5, N=500, r_mean=0.8, r_std=0.1)

#print("Alpha:", alpha)
#print("Beta:", beta)

intercept, coefficients = build_model(X, Y)

#print("Intercept:", intercept)
#print("Coefficients:", coefficients)

# ---- 2. 新しい定式化(QCQP)による最適化 ----
# 新しい定式化で必要となるパラメータを設定
p_bar = np.full(5, 0.8)  # 平均価格ベクトルp̄をr_meanで初期化
epsilon = 0.1  # 許容誤差ε

# 新しい最適化関数を呼び出し
start_2 = time.time()
optimize_price_new(
    intercept,
    coefficients,
    p_bar=p_bar,
    epsilon=epsilon
)
end_2 = time.time()
print(f"Optimization Time (New Formulation): {end_2 - start_2:.4f} seconds")
print("Optimization completed.")
print(f"New Optimization Time: {end_2 - start_2:.4f} seconds")