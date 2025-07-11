import cvxpy as cp
from cvxpy.error import DCPError 
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
) -> Tuple[NDArray, float, float]:
    """
    価格の平均値からの距離を制約とする二次制約モデルを用いて、総売上を最大化する価格を計算します。
    PDFの(3)-(4)式を実装したものです。
    時間切れの場合でも、暫定解があればその結果を返します。

    Args:
        intercepts (NDArray): 需要予測モデルの切片 (θ_0)
        coefficients (List[NDArray]): 需要予測モデルの係数 (Θ_0)
        p_bar (NDArray): 平均価格ベクトル (p̄)
        epsilon (float): 許容される誤差 (ε)

    Returns:
        Tuple[NDArray, float, float]: 最適価格、最大売上、計算時間
    """
    m = gp.Model("price_optimization_qcqp")

    num_products = len(intercepts)

    # ---- 変数の定義 ----
    prices_mvar = m.addMVar(shape=num_products, name="p", lb=0.0)

    # ---- 目的関数の設定 ----
    objective = prices_mvar @ coefficients @ prices_mvar + intercepts @ prices_mvar
    m.setObjective(objective, GRB.MAXIMIZE)

    # ---- 制約条件の設定 ----
    quad_term_wrap = prices_mvar @ prices_mvar
    lin_term_wrap  = 2 * (p_bar @ prices_mvar)
    rhs_val        = float(epsilon - (p_bar @ p_bar))

    quad_expr = quad_term_wrap.item() - lin_term_wrap.item()
    m.addQConstr(quad_expr <= rhs_val, name="norm_constraint")

    # ---- 実行時間の制限 ----
    #m.setParam(GRB.Param.OutputFlag, 0) 
    m.setParam(GRB.Param.TimeLimit, 5) # 1分

    # ---- 最適化の実行 ----
    m.optimize()
    
    # 実行時間の取得
    m_time = m.Runtime

    # ---- 結果の返却 ----
    # 最適解が得られた場合、または時間切れでも実行可能解が見つかっている場合
    if m.Status == GRB.OPTIMAL or (m.Status == GRB.TIME_LIMIT and m.SolCount > 0):
        if m.Status == GRB.TIME_LIMIT:
            print("Warning: Time limit reached. Returning the best solution found so far (suboptimal).")
        
        optimal_prices = prices_mvar.X
        max_revenue = m.ObjVal
        return optimal_prices, max_revenue, m_time
    
    # 時間切れで実行可能解が見つからなかった場合
    elif m.Status == GRB.TIME_LIMIT:
        print("Error: Time limit reached, but no feasible solution was found.")
        return np.array([]), 0.0, m_time
        
    # その他の理由で最適化が失敗した場合
    else:
        print(f"Optimization was not successful. Status code: {m.Status}")
        return np.array([]), 0.0, m_time
    

