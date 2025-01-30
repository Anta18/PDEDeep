import numpy as np


def simulate_paths_with_jumps(X0, mu, sigma, J, lambda_, N, T, M, d):
    """
    Simulate M paths of a d-dimensional jump-diffusion process over N steps with:
      - X0: initial asset prices,
      - mu: drift vector,
      - sigma: volatility vector,
      - J: jump size (can be a vector or scalar),
      - lambda_: intensity of the Poisson process.
    """
    dt = T / N
    paths = np.zeros((M, N + 1, d))
    paths[:, 0, :] = X0
    for n in range(1, N + 1):
        Z = np.random.normal(size=(M, d))
        jumps = np.random.poisson(lambda_ * dt, size=(M, d))
        jump_component = J * jumps
        paths[:, n, :] = (
            paths[:, n - 1, :]
            + mu * paths[:, n - 1, :] * dt
            + sigma * paths[:, n - 1, :] * np.sqrt(dt) * Z
            + jump_component
        )
    return paths
