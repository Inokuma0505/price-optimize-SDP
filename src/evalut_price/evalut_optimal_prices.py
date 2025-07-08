import numpy as np
from numpy.typing import NDArray
from typing import Tuple

def calculate_true_revenue(
    optimal_prices: NDArray,
    alpha_true: NDArray,
    beta_true: NDArray
) -> Tuple[float, NDArray]:
    """
    得られた最適価格を、真の需要モデルで評価し、真の総売上を計算します。

    Args:
        optimal_prices (NDArray): 最適化によって得られた価格ベクトル p*
        alpha_true (NDArray): 真の需要モデルの切片ベクトル α
        beta_true (NDArray): 真の需要モデルの係数行列 β

    Returns:
        Tuple[float, NDArray]: (真の総売上, 真の需要ベクトル)
    """
    # 真のモデルによる需要を計算: d(p*) = α - β p*
    # np.maximum(0, ...) を使い、需要がマイナスにならないようにクリップします。
    true_demand = np.maximum(0, alpha_true - (beta_true @ optimal_prices))

    # 真のモデルによる総売上を計算: R(p*) = (p*)^T * d(p*)
    true_revenue = optimal_prices @ true_demand

    return true_revenue, true_demand