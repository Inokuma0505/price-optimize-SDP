import time
import numpy as np


from src.create_data.create_data import create_data
from src.linear_regression.build_model import build_model
from src.ex_price_optimize.ex_optimize_price import optimize_price
from src.new_price_optimize.new_optimize_price import optimize_price_new
from src.new_price_optimize.new_optimize_price_sdp import optimize_price_sdp

# ---- 1. データとモデルの準備 ----
alpha, beta, X, Y = create_data(M=20, N=500, r_mean=0.8, r_std=0.1)
intercept, coefficients = build_model(X, Y)

p_bar = np.full(20, 0.8)  # 平均価格ベクトルp̄をr_meanで初期化
epsilon = 0.9  # 許容誤差ε

print("--- Optimization Start ---")

# existing optimization function

optimal_prices, max_revenue, m_time = optimize_price(
    intercept,
    coefficients,
    r_min=0.5,  # 最小価格
    r_max=1.1,  # 最大価格
)


# ---- 2. 新しい定式化(QCQP)による最適化 ----

new_prices, new_max_revenue, new_m_time = optimize_price_new(
    intercept,
    coefficients,
    p_bar=p_bar,
    epsilon=epsilon
)



# ---- 3. SDP双対問題による最適化 ----

new_prices_sdp, new_max_revenue_sdp, new_solve_time = optimize_price_sdp(
    intercept,
    coefficients,
    p_bar=p_bar,
    epsilon=epsilon
)

print("--- Existing Formulation (Gurobi) ---")
print(f"Optimal Prices: {np.round(optimal_prices, 4)}")
print(f"Maximum Revenue: {max_revenue:.4f}")
print(f"Optimization Time (Gurobi): {m_time:.4f} seconds\n")

print("--- New Formulation (QCQP) ---")
print(f"Optimal Prices: {np.round(new_prices, 4)}")
print(f"Maximum Revenue: {new_max_revenue:.4f}")
print(f"Optimization Time (QCQP with Gurobi): {new_m_time:.4f} seconds\n")

print("--- SDP Dual Formulation with Price Recovery ---")
print(f"Optimal Prices: {np.round(new_prices_sdp, 4)}")
print(f"Maximum Revenue: {new_max_revenue_sdp:.4f}")
print(f"Optimization Time (SDP with CVXPY): {new_solve_time:.4f} seconds\n")

