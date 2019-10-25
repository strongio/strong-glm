import numpy as np


def random_multivariate_normal(num_rows: int, num_dims: int) -> np.ndarray:
    """
    Random draws from a multivariate normal with a randomly chosen (unit) covariance matrix.
    """
    # generate random covariance:
    L = np.clip(np.random.laplace(size=(num_dims, num_dims)), -1.5, 1.5)
    L[range(num_dims), range(num_dims)] = np.exp(L[range(num_dims), range(num_dims)])
    cov = L @ L.T
    # standardize covariance (i.e. make diagonal all ones):
    std = np.sqrt(cov.diagonal())
    cov /= (std[..., None] @ std[..., None, :])
    return np.random.multivariate_normal(mean=np.zeros(num_dims), cov=cov, size=num_rows)


def simulate_data(num_rows: int, linear_pred_betas: np.ndarray, binary_pred_betas: np.ndarray) -> 'DataFrame':
    """
    Simulate data for a model-matrix and a response variable.
    """
    from pandas import DataFrame
    num_preds = len(linear_pred_betas) + len(binary_pred_betas)

    def label(x):
        return 'x_{}'.format(str(x).rjust(num_preds, "0"))

    X = random_multivariate_normal(num_rows, num_preds)
    df = DataFrame(index=range(num_rows))

    pred_counter = 1
    for _ in range(len(linear_pred_betas)):
        df[label(pred_counter)] = X[:, pred_counter - 1]
        pred_counter += 1
    for _ in range(len(binary_pred_betas)):
        df[label(pred_counter)] = (X[:, pred_counter - 1] > np.random.normal()).astype('float')
        pred_counter += 1

    # simulate outcome (no noise)
    betas_true = np.concatenate([linear_pred_betas, binary_pred_betas])  #
    df.insert(0, 'y', df.values @ betas_true)

    # return model-matrix & outcome
    return df
