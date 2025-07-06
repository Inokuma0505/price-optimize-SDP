import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor
from numpy.typing import NDArray
from typing import List, Tuple


def build_model(X: NDArray, y: NDArray) -> Tuple[NDArray, List[NDArray]]:
    """
    Build a multi-output linear regression model and return the intercepts and coefficients.

    Parameters:
    X (NDArray): Input features. Shape: (n_samples, n_features)
    y (NDArray): Target values. Shape: (n_samples, n_outputs)

    Returns:
    Tuple[NDArray, List[NDArray]]: 
        - Intercepts of the models (one for each output). Shape: (n_outputs,)
        - List of coefficient arrays for each model.
    """
    # Create a multi-output linear regression model
    # This wrapper creates one LinearRegression model for each target column in y
    model = MultiOutputRegressor(LinearRegression())

    # Fit the model to the data
    model.fit(X, y)
    
    # Get the intercepts and coefficients from each of the fitted estimators
    # model.estimators_ is a list of fitted LinearRegression models
    intercepts = np.array([estimator.intercept_ for estimator in model.estimators_])
    coefficients = [estimator.coef_ for estimator in model.estimators_]

    return intercepts, coefficients

