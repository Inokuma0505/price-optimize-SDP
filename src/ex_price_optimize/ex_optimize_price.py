import gurobipy as gp
from gurobipy import GRB
import numpy as np
from numpy.typing import NDArray

def optimize_price(
    intercepts: NDArray, 
    coefficients: NDArray, 
    r_min: float, 
    r_max: float
) -> tuple[NDArray, float]:
    """
    Given a demand model, find the prices that maximize total revenue.

    Parameters:
    - intercepts (NDArray): Intercepts of the linear demand model (alpha).
    - coefficients (NDArray): Coefficients of the linear demand model (beta).
    - r_min (float): Minimum allowable price.
    - r_max (float): Maximum allowable price.

    Returns:
    - tuple[NDArray, float]: A tuple containing the optimal prices and the maximum revenue.
    """
    
    m = gp.Model("price_optimization")
    
    # 1. Define variables: prices for each product
    num_products = len(intercepts)
    prices = m.addMVar(shape=num_products, name="prices", lb=r_min, ub=r_max)
    
    # 2. Define the demand based on the model
    # quantity = intercepts + coefficients @ prices
    quantity = intercepts + prices @ coefficients.T

    # 3. Set objective function: Maximize total revenue (price * quantity)
    # revenue = prices @ quantity
    total_revenue = prices @ intercepts + prices @ coefficients.T @ prices
    m.setObjective(total_revenue, GRB.MAXIMIZE)

    # Optional: Add constraint for non-negative demand
    m.addConstr(quantity >= 0, name="non_negative_demand")
    
    # 4. Solve the optimization problem
    m.optimize()
    
    if m.Status == GRB.OPTIMAL:
        optimal_prices = prices.X
        max_revenue = m.ObjVal
        return optimal_prices, max_revenue
    else:
        print("Optimization was not successful.")
        return np.array([]), 0.0