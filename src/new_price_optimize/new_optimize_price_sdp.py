import cvxpy as cp
import numpy as np
from numpy.typing import NDArray
from typing import List, Tuple

def optimize_price_sdp_mosek(
    intercepts: NDArray,
    coefficients: NDArray, # List[NDArray]からNDArrayに変更
    p_bar: NDArray,
    epsilon: float,
) -> Tuple[NDArray | None, float | None, float | None]:
    """
    QCQP（二次制約付き二次計画）問題のラグランジュ双対問題をSDP（半正定値計画）として定式化し、
    総売上の最大値と最適価格を計算します。このアプローチはS-procedureに基づいています。
    ソルバーとしてMOSEKを使用します。

    Args:
        intercepts (NDArray): 需要予測モデルの切片 (θ_0)
        coefficients (NDArray): 需要予測モデルの係数行列 (Θ_0)
        p_bar (NDArray): 平均価格ベクトル (p̄)
        epsilon (float): 許容される誤差 (ε)

    Returns:
        Tuple[NDArray | None, float | None, float | None]: (最適価格, 最大売上, 計算時間)
    """
    num_products = len(intercepts)

    # ---- CVXPYの変数を定義 ----
    mu_1 = cp.Variable(pos=True)
    mu_2 = cp.Variable()

    # ---- 行列の構築 ----
    # 目的関数 g2(p) = -(p^T*coeff*p + intercepts^T*p) の行列表現 M(g2)
    M_g2 = np.zeros((num_products + 1, num_products + 1))
    M_g2[0, 1:] = -0.5 * intercepts
    M_g2[1:, 0] = -0.5 * intercepts
    

    # SDPの定式化では、2次形式の行列は対称でなければならない
    symmetric_coefficients = 0.5 * (coefficients + coefficients.T)
    M_g2[1:, 1:] = -symmetric_coefficients
    
    # (デバッグ用に残す場合はコメントアウトを解除)
    # print(f"Shape of M_g2: {M_g2}")

    # 制約関数 g1(p) の行列表現 M(g1)
    M_g1 = np.zeros((num_products + 1, num_products + 1))
    M_g1[0, 0] = p_bar @ p_bar - epsilon
    M_g1[0, 1:] = -p_bar
    M_g1[1:, 0] = -p_bar
    M_g1[1:, 1:] = np.identity(num_products)
    # print(f"Shape of M_g1: {M_g1}")

    # 双対変数 mu_2 の行列表現 M_I
    M_I = np.zeros((num_products + 1, num_products + 1))
    M_I[0, 0] = 1
    # print(f"Shape of M_I: {M_I}")

    # ---- SDPの制約条件（LMI） ----
    LMI = M_g2 + mu_1 * M_g1 - mu_2 * M_I
    constraints = [LMI >> 0]
    problem = cp.Problem(cp.Maximize(mu_2), constraints)
    
    # ---- ソルバーをMOSEKに変更 ----
    try:
        problem.solve(solver=cp.MOSEK, verbose=False)
        solve_time = problem.solver_stats.solve_time
    except cp.error.SolverError:
        print("Solver MOSEK not found. Please ensure it is installed and licensed.")
        return None, None, None

    if problem.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
        # ---- 最適価格の復元 ----
        LMI_val = M_g2 + mu_1.value * M_g1 - mu_2.value * M_I

        eigenvalues, eigenvectors = np.linalg.eigh(LMI_val)
        p_homogeneous = eigenvectors[:, 0]


        # 固有ベクトルの符号は任意なので、p_homogeneous[0]が正になるように調整
        if p_homogeneous[0] < 0:
            p_homogeneous = -p_homogeneous

        # p_homogeneous[0] (t) がゼロに近い場合はエラーハンドリング
        if np.isclose(p_homogeneous[0], 0):
            print("Price recovery failed: homogeneous scaling factor is close to zero.")
            return None, None, solve_time

        optimal_prices = p_homogeneous[1:] / p_homogeneous[0]
        max_revenue = -problem.value

        print("--- SDP Dual Formulation with Price Recovery (MOSEK) ---")
        print(f"Optimal Prices: {np.round(optimal_prices, 4)}")
        print(f"Maximum Revenue: {max_revenue:.4f}")
        print(f"Solve Time: {solve_time:.6f} seconds")

        return optimal_prices, max_revenue, solve_time
    else:
        print("SDP Optimization was not successful.")
        print(f"Problem status: {problem.status}")
        return None, None, None