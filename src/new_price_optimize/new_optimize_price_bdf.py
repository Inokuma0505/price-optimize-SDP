import gurobipy as gp
from gurobipy import GRB
import numpy as np
from numpy.typing import NDArray
from typing import Tuple

def optimize_price_new_bdf(
    intercepts: NDArray,
    coefficients: NDArray,
    p_bar: NDArray,
    epsilon: float,
) -> Tuple[NDArray | None, float | None, float | None]:
    """
    総売上を最大化する価格を計算します。
    制約条件を ||p - p̄||₂² < ε の形で直接的にモデル化します。
    PDFの(1)-(2)式を実装したものです。

    Args:
        intercepts (NDArray): 需要予測モデルの切片 (θ_0)
        coefficients (NDArray): 需要予測モデルの係数 (Θ_0)
        p_bar (NDArray): 平均価格ベクトル (p̄)
        epsilon (float): 許容される誤差 (ε)

    Returns:
        Tuple[NDArray | None, float | None, float | None]: (最適価格, 最大売上, 計算時間)
    """
    try:
        m = gp.Model("price_optimization_qcqp_bdf")

        num_products = len(intercepts)

        # ---- 変数の定義 ----
        # 価格は0以上とする
        prices_mvar = m.addMVar(shape=num_products, name="p", lb=0.0)

        # ---- 目的関数の設定 ----
        # 目的関数: p^T * Θ₀ * p + θ₀^T * p
        objective = prices_mvar @ coefficients @ prices_mvar + intercepts @ prices_mvar
        m.setObjective(objective, GRB.MAXIMIZE)

        # ---- 制約条件の設定 ----
        # ||p - p̄||₂² <= ε を直接モデル化
        delta_p = prices_mvar - p_bar
        norm_squared_expr = delta_p @ delta_p
        m.addQConstr(norm_squared_expr.item() <= epsilon, name="norm_constraint")

        # ---- 最適化の実行 ----
        m.optimize()
        
        solve_time = m.Runtime

        # ---- 結果の返却 ----
        if m.Status == GRB.OPTIMAL:
            optimal_prices = prices_mvar.X
            max_revenue = m.ObjVal
            return optimal_prices, max_revenue, solve_time
        else:
            print("Optimization was not successful.")
            print(f"Gurobi status code: {m.Status}")
            return None, None, solve_time

    except gp.GurobiError as e:
        print(f"Error code {e.errno}: {e}")
        return None, None, None