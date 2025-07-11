import json
import numpy as np
import pandas as pd
from tqdm import tqdm  # 進捗バーを表示するためにtqdmをインポート

# srcディレクトリから必要なモジュールをインポート
from src.create_data.create_data import create_data
from src.linear_regression.build_model import build_model
from src.linear_regression.build_model import build_model
from src.new_price_optimize.new_optimize_price import optimize_price_new, optimize_price_new_mosek
from src.new_price_optimize.new_optimize_price_bdf import optimize_price_new_bdf
from src.new_price_optimize.new_optimize_price_sdp import optimize_price_sdp_mosek

# --- 評価用関数の定義 ---
def evaluate_true_revenue(prices, true_alpha, true_beta):
    """
    与えられた価格ベクトルに基づき、真の需要関数 (alpha, beta) を用いて
    総売上を計算（評価）します。
    """
    if prices is None:
        return None
    true_demand = true_alpha + true_beta @ prices
    true_demand[true_demand < 0] = 0
    total_revenue = prices @ true_demand
    return float(total_revenue)

# --- メインの実験ロジック ---
def run_experiments(num_experiments):
    """
    指定された回数の価格最適化実験を実行し、結果をリストとして返す。
    """
    all_results = []
    
    # tqdmを使ってループの進捗を表示
    for i in tqdm(range(num_experiments), desc="Running Experiments"):
        # 乱数シードを固定して、各実験の再現性を確保
        np.random.seed(i)
        
        # --- 1. 真の需要モデルと学習用データの生成 ---
        num_products = 20
        num_samples = 500
        alpha, beta, X, Y = create_data(M=num_products, N=num_samples, r_mean=0.8, r_std=0.1, delta=0)

        # --- 2. 線形回帰による需要モデルの推定 ---
        est_intercepts, est_coefficients = build_model(X, Y)


        # --- 3. 最適化の共通パラメータ ---
        p_bar = np.full(20, 0.8)  # 平均価格ベクトルp̄をr_meanで初期化
        epsilon = 0.09 * 20  # 許容誤差ε

        # --- 4. 各手法による最適化の実行と評価 ---
        
        # a) Gurobi (推定値)
        gurobi_prices, gurobi_rev, runtime_gurobi = optimize_price_new(
            est_intercepts, est_coefficients, p_bar, epsilon
        )
        gurobi_eval_rev = evaluate_true_revenue(gurobi_prices, alpha, beta)

        # b) MOSEK QCQP (推定値)
        mosek_qcqp_prices, mosek_qcqp_rev, runtime_mosek_qcqp = optimize_price_new_mosek(
            est_intercepts, est_coefficients, p_bar, epsilon
        )
        mosek_qcqp_eval_rev = evaluate_true_revenue(mosek_qcqp_prices, alpha, beta)

        # c) MOSEK SDP (推定値)
        sdp_prices, sdp_rev, runtime_sdp = optimize_price_sdp_mosek(
            est_intercepts, est_coefficients, p_bar, epsilon
        )
        sdp_eval_rev = evaluate_true_revenue(sdp_prices, alpha, beta)

        # d) 真の最適解 (Gurobi使用)
        true_optimal_prices, true_optimal_rev, runtime_true_optimal = optimize_price_new(
            alpha, beta, p_bar, epsilon
        )
        
        # --- 5. 結果の格納 ---
        experiment_data = {
            "number":i,
            "true_alpha": alpha.tolist(),
            "true_beta": beta.tolist(),
            "estimated_intercepts": est_intercepts.tolist(),
            "estimated_coefficients": est_coefficients.tolist(),
            "true_optimal_solution": {
                "prices": true_optimal_prices.tolist() if true_optimal_prices is not None else None,
                "revenue": true_optimal_rev,
            },
            "gurobi_qcqp_result": {
                "optimal_prices": gurobi_prices.tolist() if gurobi_prices is not None else None,
                "optimal_revenue": gurobi_rev,
                "evaluated_revenue": gurobi_eval_rev,
                "runtime": runtime_gurobi,
            },
            "mosek_qcqp_result": {
                "optimal_prices": mosek_qcqp_prices.tolist() if mosek_qcqp_prices is not None else None,
                "optimal_revenue": mosek_qcqp_rev,
                "evaluated_revenue": mosek_qcqp_eval_rev,
                "runtime": runtime_mosek_qcqp,
            },
            "mosek_sdp_result": {
                "optimal_prices": sdp_prices.tolist() if sdp_prices is not None else None,
                "optimal_revenue": sdp_rev,
                "evaluated_revenue": sdp_eval_rev,
                "runtime": runtime_sdp,
            }
        }
        all_results.append(experiment_data)
        
    return all_results

# --- スクリプト実行部分 ---
if __name__ == "__main__":
    # 実験を実行
    results = run_experiments(num_experiments=100)
    
    # 結果をJSONファイルに出力
    output_filename = "experiment_results_delta0.json"
    with open(output_filename, "w") as f:
        json.dump(results, f, indent=4)
        
    print(f"Successfully completed 100 experiments. Results are saved to '{output_filename}'.")