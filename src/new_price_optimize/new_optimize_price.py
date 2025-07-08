import cvxpy as cp
import gurobipy as gp
from gurobipy import GRB
import numpy as np
from numpy.typing import NDArray
from typing import List, Tuple

def optimize_price_new(
    intercepts: NDArray,
    coefficients: List[NDArray],
    p_bar: NDArray,
    epsilon: float,
) -> tuple[NDArray, float]:
    """
    価格の平均値からの距離を制約とする二次制約モデルを用いて、総売上を最大化する価格を計算します。
    PDFの(3)-(4)式を実装したものです。

    Args:
        intercepts (NDArray): 需要予測モデルの切片 (θ_0)
        coefficients (List[NDArray]): 需要予測モデルの係数 (Θ_0)
        p_bar (NDArray): 平均価格ベクトル (p̄)
        epsilon (float): 許容される誤差 (ε)

    Returns:
        tuple[NDArray, float]: 最適価格と最大売上
    """
    m = gp.Model("price_optimization_qcqp")

    num_products = len(intercepts)

    # ---- 変数の定義 ----
    prices_mvar = m.addMVar(shape=num_products, name="p", lb=0.0)

    # ---- 目的関数の設定 ----
    objective = prices_mvar @ coefficients @ prices_mvar + intercepts @ prices_mvar
    m.setObjective(objective, GRB.MAXIMIZE)

    # ---- 制約条件の設定（日本語版） ----
    # MVar で作った 0-d ラッパーから中身を取り出す
    quad_term_wrap = prices_mvar @ prices_mvar          # MQuadExpr, shape=()
    lin_term_wrap  = 2 * (p_bar @ prices_mvar)         # MLinExpr, shape=()
    rhs_val         = float(epsilon - (p_bar @ p_bar))  # Python の float に変換

    # .item() で純粋な QuadExpr／LinExpr を取得
    quad_expr = quad_term_wrap.item() - lin_term_wrap.item()

    # そのまま <= 演算子で制約を追加
    m.addQConstr(quad_expr <= rhs_val, name="norm_constraint")



    # ---- 最適化の実行 ----
    m.optimize()
    
    # 実行時間の取得
    m_time = m.Runtime

    # ---- 結果の返却 ----
    if m.Status == GRB.OPTIMAL:
        optimal_prices = prices_mvar.X
        max_revenue = m.ObjVal
        #print("--- New Formulation (QCQP) ---")
        #print(f"Optimal Prices: {np.round(optimal_prices, 4)}")
        #print(f"Maximum Revenue: {max_revenue:.4f}")
        return optimal_prices, max_revenue, m_time
    else:
        print("Optimization was not successful.")
        return np.array([]), 0.0, 0.0
    

def optimize_price_new_mosek(
    intercepts: NDArray,
    coefficients: NDArray,
    p_bar: NDArray,
    epsilon: float,
) -> Tuple[NDArray | None, float | None, float | None]:
    """
    価格の平均値からの距離を制約とするQCQPモデルを用いて、総売上を最大化します。
    問題をCVXPYで記述し、MOSEKソルバーを使用して解きます。
    (ValueError: Quadratic form matrices must be symmetric/Hermitian. 対策済み)

    Args:
        intercepts (NDArray): 需要予測モデルの切片 (θ_0)
        coefficients (NDArray): 需要予測モデルの係数 (Θ_2)
        p_bar (NDArray): 平均価格ベクトル (p̄)
        epsilon (float): 許容される誤差 (ε)

    Returns:
        Tuple[NDArray | None, float | None, float | None]: (最適価格, 最大売上, 計算時間)
    """
    num_products = len(intercepts)

    # ---- CVXPYの変数を定義 ----
    prices = cp.Variable(num_products, name="p", pos=True)

    # ---- 【修正点】係数行列を対称行列に変換 ----
    # cvxpy.quad_formは対称行列を要求するため、(Q + Q.T)/2 を計算して対称化します。
    # この処理を行っても、二次形式の値 (p^T * Q * p) は変わりません。
    Q_sym = 0.5 * (coefficients + coefficients.T)

    # ---- 目的関数の設定 ----
    # 対称化された行列 Q_sym を使用します
    objective = cp.Maximize(cp.quad_form(prices, Q_sym) + intercepts @ prices)

    # ---- 制約条件の設定 ----
    # 制約: ||p - p̄||^2 <= ε
    constraints = [cp.sum_squares(prices - p_bar) <= epsilon]

    # ---- 問題の定義と最適化の実行 ----
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.MOSEK)

    # 実行時間を取得
    solve_time = problem.solver_stats.solve_time

    # ---- 結果の返却 ----
    if problem.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
        optimal_prices = prices.value
        max_revenue = problem.value
        return optimal_prices, max_revenue, solve_time
    else:
        print("QCQP Optimization with MOSEK was not successful.")
        print(f"Problem status: {problem.status}")
        return None, None, None