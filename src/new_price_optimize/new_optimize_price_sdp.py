import cvxpy as cp
import numpy as np
from numpy.typing import NDArray
from typing import List, Tuple

def optimize_price_sdp_mosek(
    intercepts: NDArray,
    coefficients: List[NDArray],
    p_bar: NDArray,
    epsilon: float,
) -> Tuple[NDArray | None, float | None, float | None]:
    """
    QCQP（二次制約付き二次計画）問題のラグランジュ双対問題をSDP（半正定値計画）として定式化し、
    総売上の最大値と最適価格を計算します。このアプローチはS-procedureに基づいています。
    ソルバーとしてMOSEKを使用します。

    Args:
        intercepts (NDArray): 需要予測モデルの切片 (θ_0)
        coefficients (List[NDArray]): 需要予測モデルの係数 (Θ_2)
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
    # 目的関数 g2(p) = -p^T*coeff*p - intercepts^T*p の行列表現 M(g2)
    M_g2 = np.zeros((num_products + 1, num_products + 1))
    M_g2[0, 1:] = -0.5 * intercepts
    M_g2[1:, 0] = -0.5 * intercepts
    M_g2[1:, 1:] = -coefficients

    # 制約関数 g1(p) の行列表現 M(g1)
    M_g1 = np.zeros((num_products + 1, num_products + 1))
    M_g1[0, 0] = p_bar @ p_bar - epsilon
    M_g1[0, 1:] = -p_bar.T
    M_g1[1:, 0] = -p_bar
    M_g1[1:, 1:] = np.identity(num_products)

    # 双対変数 mu_2 の行列表現 M_I
    M_I = np.zeros((num_products + 1, num_products + 1))
    M_I[0, 0] = 1

    # ---- SDPの制約条件（LMI） ----
    LMI = M_g2 + mu_1 * M_g1 - mu_2 * M_I

    constraints = [LMI >> 0]
    problem = cp.Problem(cp.Maximize(mu_2), constraints)
    
    # ---- ソルバーをMOSEKに変更 ----
    problem.solve(solver=cp.MOSEK)

    # 実行時間を取得
    solve_time = problem.solver_stats.solve_time

    if problem.status == cp.OPTIMAL or problem.status == cp.OPTIMAL_INACCURATE:
        # ---- 最適価格の復元 ----
        # 最適点ではLMIは特異になる。その核ベクトルを求める。
        # LMIの行列を、得られた最適変数 mu_1, mu_2 の値で構築
        LMI_val = M_g2 + mu_1.value * M_g1 - mu_2.value * M_I

        # 最小固有値に対応する固有ベクトルを計算
        eigenvalues, eigenvectors = np.linalg.eigh(LMI_val)
        # 固有ベクトルは列ベクトルなので、最小固有値に対応する最初の列を取得
        p_homogeneous = eigenvectors[:, 0]

        # 固有ベクトルをスケール調整（均質化されているため）
        # p_homogeneous = [t, p_1*t, p_2*t, ...]
        # t = p_homogeneous[0] で割ることで、価格ベクトル p を得る
        optimal_prices = p_homogeneous[1:] / p_homogeneous[0]

        # 最大売上
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

# ---- 以下は使用例です ----
if __name__ == '__main__':
    # MOSEKがインストールされているか確認
    if 'MOSEK' in cp.installed_solvers():
        print("MOSEK solver is available.\n")
        
        # ダミーデータの作成
        num_products = 5
        np.random.seed(0)
        intercepts = np.random.rand(num_products) * 10
        # 係数行列を負定値に近づけるための処理
        A = np.random.rand(num_products, num_products)
        coefficients = - (A.T @ A) 
        p_bar = np.random.rand(num_products) * 20
        epsilon = 50.0

        print("--- Input Data ---")
        print(f"Number of products: {num_products}")
        print(f"Epsilon: {epsilon}\n")

        # 最適化の実行
        optimize_price_sdp_mosek(intercepts, coefficients, p_bar, epsilon)
    else:
        print("MOSEK solver is not installed or not found by CVXPY.")
        print("Please install MOSEK and its Python interface.")
        print("Installation instructions can be found at: https://www.mosek.com/documentation/")