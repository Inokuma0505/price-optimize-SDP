from src.create_data.create_data import create_data
from src.linear_regression.build_model import build_model

alpha, beta, X, Y = create_data(M=5, N=100, r_mean=0.8, r_std=0.1)

print("Alpha:", alpha)
print("Beta:", beta)

intercept, coefficients = build_model(X, Y)

print("Intercept:", intercept)
print("Coefficients:", coefficients)



