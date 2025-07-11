import time
import numpy as np


from src.create_data.create_data import create_data
from src.linear_regression.build_model import build_model
from src.ex_price_optimize.ex_optimize_price import optimize_price
from src.new_price_optimize.new_optimize_price import optimize_price_new, optimize_price_new_mosek
from src.new_price_optimize.new_optimize_price_bdf import optimize_price_new_bdf
from src.new_price_optimize.new_optimize_price_sdp import optimize_price_sdp_mosek

# ---- 1. データとモデルの準備 ----
alpha, beta, X, Y = create_data(M=20, N=500, r_mean=0.8, r_std=0.1,delta=0)
intercept, coefficients = build_model(X, Y)

print(coefficients)

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

# optimize_price_new_mosekを使用して、MOSEKを利用したQCQPの最適化
new_prices_mosek, new_max_revenue_mosek, new_m_time_mosek = optimize_price_new_mosek(
    intercept,
    coefficients,
    p_bar=p_bar,
    epsilon=epsilon
)

# ---- 3. SDP双対問題による最適化 ----

new_prices_sdp, new_max_revenue_sdp, new_solve_time = optimize_price_sdp_mosek(
    intercept,
    coefficients,
    p_bar=p_bar,
    epsilon=epsilon
)



print("QCQP Gurobi")
print(new_prices)
print(new_max_revenue)
print(new_m_time)

print("QCQP MOSEK")
print(new_prices_mosek)
print(new_max_revenue_mosek)
print(new_m_time_mosek)

print("SDP")
print(new_prices_sdp)
print(new_max_revenue_sdp)
print(new_solve_time)

# 結果が一致しているかの比較
# QCQPとSDPの最適解が一致しているか確認
if np.allclose(new_prices, new_prices_sdp, atol=1e-3):
    print("QCQP and SDP optimal prices match within tolerance.")
else:
    print("QCQP and SDP optimal prices do not match.")
    
# QCQPとBDFの最適解が一致しているか確認
if np.allclose(new_prices, new_prices_mosek, atol=1e-3):
    print("QCQP and BDF optimal prices match within tolerance.")
else:
    print("QCQP and BDF optimal prices do not match.")
    


