import numpy as np
import tensorflow as tf
from simulation import simulate_paths_with_jumps


class Equation(object):
    """Base class for defining PDE related function."""

    def __init__(self, eqn_config):
        self.dim = eqn_config.dim
        self.total_time = eqn_config.total_time
        self.num_time_interval = eqn_config.num_time_interval
        self.delta_t = self.total_time / self.num_time_interval
        self.sqrt_delta_t = np.sqrt(self.delta_t)
        self.y_init = None

    def sample(self, num_sample):
        """Sample forward SDE."""
        raise NotImplementedError

    def f_tf(self, t, x, y, z):
        """Generator function in the PDE."""
        raise NotImplementedError

    def g_tf(self, t, x):
        """Terminal condition of the PDE."""
        raise NotImplementedError


class HJBLQ(Equation):
    """HJB equation in PNAS paper doi.org/10.1073/pnas.1718942115"""

    def __init__(self, eqn_config):
        super(HJBLQ, self).__init__(eqn_config)
        self.x_init = np.zeros(self.dim)
        self.sigma = np.sqrt(2.0)
        self.lambd = 1.0

    def sample(self, num_sample):
        dw_sample = (
            np.random.normal(size=[num_sample, self.dim, self.num_time_interval])
            * self.sqrt_delta_t
        )
        x_sample = np.zeros([num_sample, self.dim, self.num_time_interval + 1])
        x_sample[:, :, 0] = np.ones([num_sample, self.dim]) * self.x_init
        for i in range(self.num_time_interval):
            x_sample[:, :, i + 1] = x_sample[:, :, i] + self.sigma * dw_sample[:, :, i]
        return dw_sample, x_sample

    def f_tf(self, t, x, y, z):
        return -self.lambd * tf.reduce_sum(tf.square(z), 1, keepdims=True) / 2

    def g_tf(self, t, x):
        return tf.math.log((1 + tf.reduce_sum(tf.square(x), 1, keepdims=True)) / 2)


class AllenCahn(Equation):
    """Allen-Cahn equation in PNAS paper doi.org/10.1073/pnas.1718942115"""

    def __init__(self, eqn_config):
        super(AllenCahn, self).__init__(eqn_config)
        self.x_init = np.zeros(self.dim)
        self.sigma = np.sqrt(2.0)

    def sample(self, num_sample):
        dw_sample = (
            np.random.normal(size=[num_sample, self.dim, self.num_time_interval])
            * self.sqrt_delta_t
        )
        x_sample = np.zeros([num_sample, self.dim, self.num_time_interval + 1])
        x_sample[:, :, 0] = np.ones([num_sample, self.dim]) * self.x_init
        for i in range(self.num_time_interval):
            x_sample[:, :, i + 1] = x_sample[:, :, i] + self.sigma * dw_sample[:, :, i]
        return dw_sample, x_sample

    def f_tf(self, t, x, y, z):
        return y - tf.pow(y, 3)

    def g_tf(self, t, x):
        return 0.5 / (1 + 0.2 * tf.reduce_sum(tf.square(x), 1, keepdims=True))


class PricingDefaultRisk(Equation):
    """
    Nonlinear Black-Scholes equation with default risk in PNAS paper
    doi.org/10.1073/pnas.1718942115
    """

    def __init__(self, eqn_config):
        super(PricingDefaultRisk, self).__init__(eqn_config)
        self.x_init = np.ones(self.dim) * 100.0
        self.sigma = 0.2
        self.rate = 0.02  # interest rate R
        self.delta = 2.0 / 3
        self.gammah = 0.2
        self.gammal = 0.02
        self.mu_bar = 0.02
        self.vh = 50.0
        self.vl = 70.0
        self.slope = (self.gammah - self.gammal) / (self.vh - self.vl)

    def sample(self, num_sample):
        dw_sample = (
            np.random.normal(size=[num_sample, self.dim, self.num_time_interval])
            * self.sqrt_delta_t
        )
        x_sample = np.zeros([num_sample, self.dim, self.num_time_interval + 1])
        x_sample[:, :, 0] = np.ones([num_sample, self.dim]) * self.x_init
        for i in range(self.num_time_interval):
            x_sample[:, :, i + 1] = (1 + self.mu_bar * self.delta_t) * x_sample[
                :, :, i
            ] + (self.sigma * x_sample[:, :, i] * dw_sample[:, :, i])
        return dw_sample, x_sample

    def f_tf(self, t, x, y, z):
        piecewise_linear = (
            tf.nn.relu(tf.nn.relu(y - self.vh) * self.slope + self.gammah - self.gammal)
            + self.gammal
        )
        return (-(1 - self.delta) * piecewise_linear - self.rate) * y

    def g_tf(self, t, x):
        return tf.reduce_min(x, 1, keepdims=True)


class PricingDiffRate(Equation):
    """
    Nonlinear Black-Scholes equation with different interest rates for borrowing and lending
    in Section 4.4 of Comm. Math. Stat. paper doi.org/10.1007/s40304-017-0117-6
    """

    def __init__(self, eqn_config):
        super(PricingDiffRate, self).__init__(eqn_config)
        self.x_init = np.ones(self.dim) * 100
        self.sigma = 0.2
        self.mu_bar = 0.06
        self.rl = 0.04
        self.rb = 0.06
        self.alpha = 1.0 / self.dim

    def sample(self, num_sample):
        dw_sample = (
            np.random.normal(size=[num_sample, self.dim, self.num_time_interval])
            * self.sqrt_delta_t
        )
        x_sample = np.zeros([num_sample, self.dim, self.num_time_interval + 1])
        x_sample[:, :, 0] = np.ones([num_sample, self.dim]) * self.x_init
        factor = np.exp((self.mu_bar - (self.sigma**2) / 2) * self.delta_t)
        for i in range(self.num_time_interval):
            x_sample[:, :, i + 1] = (
                factor * np.exp(self.sigma * dw_sample[:, :, i])
            ) * x_sample[:, :, i]
        return dw_sample, x_sample

    def f_tf(self, t, x, y, z):
        temp = tf.reduce_sum(z, 1, keepdims=True) / self.sigma
        return (
            -self.rl * y
            - (self.mu_bar - self.rl) * temp
            + ((self.rb - self.rl) * tf.maximum(temp - y, 0))
        )

    def g_tf(self, t, x):
        temp = tf.reduce_max(x, 1, keepdims=True)
        return tf.maximum(temp - 120, 0) - 2 * tf.maximum(temp - 150, 0)


class BurgersType(Equation):
    """
    Multidimensional Burgers-type PDE in Section 4.5 of Comm. Math. Stat. paper
    doi.org/10.1007/s40304-017-0117-6
    """

    def __init__(self, eqn_config):
        super(BurgersType, self).__init__(eqn_config)
        self.x_init = np.zeros(self.dim)
        self.y_init = 1 - 1.0 / (1 + np.exp(0 + np.sum(self.x_init) / self.dim))
        self.sigma = self.dim + 0.0

    def sample(self, num_sample):
        dw_sample = (
            np.random.normal(size=[num_sample, self.dim, self.num_time_interval])
            * self.sqrt_delta_t
        )
        x_sample = np.zeros([num_sample, self.dim, self.num_time_interval + 1])
        x_sample[:, :, 0] = np.ones([num_sample, self.dim]) * self.x_init
        for i in range(self.num_time_interval):
            x_sample[:, :, i + 1] = x_sample[:, :, i] + self.sigma * dw_sample[:, :, i]
        return dw_sample, x_sample

    def f_tf(self, t, x, y, z):
        return (y - (2 + self.dim) / 2.0 / self.dim) * tf.reduce_sum(
            z, 1, keepdims=True
        )

    def g_tf(self, t, x):
        return 1 - 1.0 / (1 + tf.exp(t + tf.reduce_sum(x, 1, keepdims=True) / self.dim))


class QuadraticGradient(Equation):
    """
    An example PDE with quadratically growing derivatives in Section 4.6 of Comm. Math. Stat. paper
    doi.org/10.1007/s40304-017-0117-6
    """

    def __init__(self, eqn_config):
        super(QuadraticGradient, self).__init__(eqn_config)
        self.alpha = 0.4
        self.x_init = np.zeros(self.dim)
        base = self.total_time + np.sum(np.square(self.x_init) / self.dim)
        self.y_init = np.sin(np.power(base, self.alpha))

    def sample(self, num_sample):
        dw_sample = (
            np.random.normal(size=[num_sample, self.dim, self.num_time_interval])
            * self.sqrt_delta_t
        )
        x_sample = np.zeros([num_sample, self.dim, self.num_time_interval + 1])
        x_sample[:, :, 0] = np.ones([num_sample, self.dim]) * self.x_init
        for i in range(self.num_time_interval):
            x_sample[:, :, i + 1] = x_sample[:, :, i] + dw_sample[:, :, i]
        return dw_sample, x_sample

    def f_tf(self, t, x, y, z):
        x_square = tf.reduce_sum(tf.square(x), 1, keepdims=True)
        base = self.total_time - t + x_square / self.dim
        base_alpha = tf.pow(base, self.alpha)
        derivative = self.alpha * tf.pow(base, self.alpha - 1) * tf.cos(base_alpha)
        term1 = tf.reduce_sum(tf.square(z), 1, keepdims=True)
        term2 = -4.0 * (derivative**2) * x_square / (self.dim**2)
        term3 = derivative
        term4 = -0.5 * (
            2.0 * derivative
            + 4.0
            / (self.dim**2)
            * x_square
            * self.alpha
            * (
                (self.alpha - 1) * tf.pow(base, self.alpha - 2) * tf.cos(base_alpha)
                - (self.alpha * tf.pow(base, 2 * self.alpha - 2) * tf.sin(base_alpha))
            )
        )
        return term1 + term2 + term3 + term4

    def g_tf(self, t, x):
        return tf.sin(
            tf.pow(tf.reduce_sum(tf.square(x), 1, keepdims=True) / self.dim, self.alpha)
        )


class ReactionDiffusion(Equation):
    """
    Time-dependent reaction-diffusion-type example PDE in Section 4.7 of Comm. Math. Stat. paper
    doi.org/10.1007/s40304-017-0117-6
    """

    def __init__(self, eqn_config):
        super(ReactionDiffusion, self).__init__(eqn_config)
        self._kappa = 0.6
        self.lambd = 1 / np.sqrt(self.dim)
        self.x_init = np.zeros(self.dim)
        self.y_init = (
            1
            + self._kappa
            + np.sin(self.lambd * np.sum(self.x_init))
            * np.exp(-self.lambd * self.lambd * self.dim * self.total_time / 2)
        )

    def sample(self, num_sample):
        dw_sample = (
            np.random.normal(size=[num_sample, self.dim, self.num_time_interval])
            * self.sqrt_delta_t
        )
        x_sample = np.zeros([num_sample, self.dim, self.num_time_interval + 1])
        x_sample[:, :, 0] = np.ones([num_sample, self.dim]) * self.x_init
        for i in range(self.num_time_interval):
            x_sample[:, :, i + 1] = x_sample[:, :, i] + dw_sample[:, :, i]
        return dw_sample, x_sample

    def f_tf(self, t, x, y, z):
        exp_term = tf.exp((self.lambd**2) * self.dim * (t - self.total_time) / 2)
        sin_term = tf.sin(self.lambd * tf.reduce_sum(x, 1, keepdims=True))
        temp = y - self._kappa - 1 - sin_term * exp_term
        return tf.minimum(tf.constant(1.0, dtype=tf.float64), tf.square(temp))

    def g_tf(self, t, x):
        return 1 + self._kappa + tf.sin(self.lambd * tf.reduce_sum(x, 1, keepdims=True))


class HeatEquation(Equation):
    """
    A simple heat equation as an example.
    PDE: u_t + (1/2)*Δu = 0 with a given terminal condition u(T,x) = g(x).
    """

    def __init__(self, eqn_config):
        super(HeatEquation, self).__init__(eqn_config)
        self.x_init = np.zeros(self.dim)
        self.sigma = np.sqrt(2.0)  # For standard heat diffusion

    def sample(self, num_sample):
        # Simulate paths for a standard Brownian motion (corresponding to heat equation)
        dw_sample = (
            np.random.normal(size=[num_sample, self.dim, self.num_time_interval])
            * self.sqrt_delta_t
        )
        x_sample = np.zeros([num_sample, self.dim, self.num_time_interval + 1])
        x_sample[:, :, 0] = np.ones([num_sample, self.dim]) * self.x_init
        for i in range(self.num_time_interval):
            x_sample[:, :, i + 1] = x_sample[:, :, i] + self.sigma * dw_sample[:, :, i]
        return dw_sample, x_sample

    def f_tf(self, t, x, y, z):
        # For the linear heat equation, the generator f is zero.
        return tf.zeros_like(y)

    def g_tf(self, t, x):
        # Define a terminal condition, for example: u(T,x) = exp(-||x||^2)
        return tf.exp(-tf.reduce_sum(tf.square(x), axis=1, keepdims=True))


class BasketOption(Equation):
    """
    European Basket Call Option PDE:
    Under risk-neutral pricing, the PDE for a European call option on the maximum of d assets is
    given by
        u_t + 0.5 * sum_{i=1}^d sigma^2 x_i^2 u_{x_i x_i} + r sum_{i=1}^d x_i u_{x_i} - r u = 0,
    with terminal condition u(T,x) = max(max_i(x_i) - K, 0),
    where r is the risk-free rate, sigma is the volatility, and K is the strike price.
    """

    def __init__(self, eqn_config):
        super(BasketOption, self).__init__(eqn_config)
        self.dim = eqn_config.dim
        self.total_time = eqn_config.total_time
        self.num_time_interval = eqn_config.num_time_interval
        self.delta_t = self.total_time / self.num_time_interval
        self.sqrt_delta_t = np.sqrt(self.delta_t)
        self.x_init = np.ones(self.dim) * 100.0  # initial asset prices
        self.sigma = 0.2
        self.r = 0.05
        self.K = 100.0  # strike price

    def sample(self, num_sample):
        # Simulate Geometric Brownian Motion for each asset
        dw_sample = (
            np.random.normal(size=[num_sample, self.dim, self.num_time_interval])
            * self.sqrt_delta_t
        )
        x_sample = np.zeros([num_sample, self.dim, self.num_time_interval + 1])
        x_sample[:, :, 0] = self.x_init  # starting values for all samples

        drift = (self.r - 0.5 * self.sigma**2) * self.delta_t
        diffusion = self.sigma

        for i in range(self.num_time_interval):
            x_sample[:, :, i + 1] = x_sample[:, :, i] * np.exp(
                drift + diffusion * dw_sample[:, :, i]
            )
        return dw_sample, x_sample

    def f_tf(self, t, x, y, z):
        # Under risk-neutral dynamics, the generator for linear pricing PDEs is -r*y.
        return -self.r * y

    def g_tf(self, t, x):
        # Payoff for a European max-call option: max(max_i(x_i) - K, 0)
        max_x = tf.reduce_max(x, axis=1, keepdims=True)
        return tf.maximum(max_x - self.K, 0)


class BasketPut(Equation):
    """
    European Basket Put Option PDE:
    Under risk-neutral pricing, the PDE for a European put option on the minimum of d assets is
    given by
        u_t + 0.5 * sum_{i=1}^d sigma^2 x_i^2 u_{x_i x_i} + r sum_{i=1}^d x_i u_{x_i} - r u = 0,
    with terminal condition u(T,x) = max(K - min_i(x_i), 0),
    where r is the risk-free rate, sigma is the volatility, and K is the strike price.
    """

    def __init__(self, eqn_config):
        super(BasketPut, self).__init__(eqn_config)
        self.dim = eqn_config.dim
        self.total_time = eqn_config.total_time
        self.num_time_interval = eqn_config.num_time_interval
        self.delta_t = self.total_time / self.num_time_interval
        self.sqrt_delta_t = np.sqrt(self.delta_t)
        self.x_init = np.ones(self.dim) * 100.0
        self.sigma = 0.2
        self.r = 0.05
        self.K = 100.0

    def sample(self, num_sample):
        dw_sample = (
            np.random.normal(size=[num_sample, self.dim, self.num_time_interval])
            * self.sqrt_delta_t
        )
        x_sample = np.zeros([num_sample, self.dim, self.num_time_interval + 1])
        x_sample[:, :, 0] = self.x_init

        drift = (self.r - 0.5 * self.sigma**2) * self.delta_t
        diffusion = self.sigma

        for i in range(self.num_time_interval):
            x_sample[:, :, i + 1] = x_sample[:, :, i] * np.exp(
                drift + diffusion * dw_sample[:, :, i]
            )
        return dw_sample, x_sample

    def f_tf(self, t, x, y, z):
        return -self.r * y

    def g_tf(self, t, x):
        min_x = tf.reduce_min(x, axis=1, keepdims=True)
        return tf.maximum(self.K - min_x, 0)


class SumCallOption(Equation):
    """
    European Call Option on the sum of assets PDE:
    Under risk-neutral pricing, the PDE for a European call option on the sum of d assets is
    given by
        u_t + 0.5 * sum_{i=1}^d sigma^2 x_i^2 u_{x_i x_i} + r sum_{i=1}^d x_i u_{x_i} - r u = 0,
    with terminal condition u(T,x) = max(sum_i(x_i) - K, 0).
    """

    def __init__(self, eqn_config):
        super(SumCallOption, self).__init__(eqn_config)
        self.dim = eqn_config.dim
        self.total_time = eqn_config.total_time
        self.num_time_interval = eqn_config.num_time_interval
        self.delta_t = self.total_time / self.num_time_interval
        self.sqrt_delta_t = np.sqrt(self.delta_t)
        self.x_init = np.ones(self.dim) * 100.0
        self.sigma = 0.2
        self.r = 0.05
        self.K = 100.0

    def sample(self, num_sample):
        dw_sample = (
            np.random.normal(size=[num_sample, self.dim, self.num_time_interval])
            * self.sqrt_delta_t
        )
        x_sample = np.zeros([num_sample, self.dim, self.num_time_interval + 1])
        x_sample[:, :, 0] = self.x_init

        drift = (self.r - 0.5 * self.sigma**2) * self.delta_t
        diffusion = self.sigma

        for i in range(self.num_time_interval):
            x_sample[:, :, i + 1] = x_sample[:, :, i] * np.exp(
                drift + diffusion * dw_sample[:, :, i]
            )
        return dw_sample, x_sample

    def f_tf(self, t, x, y, z):
        return -self.r * y

    def g_tf(self, t, x):
        sum_x = tf.reduce_sum(x, axis=1, keepdims=True)
        return tf.maximum(sum_x - self.K, 0)


class JumpBasketPut(Equation):
    """
    Basket Put Option PDE with jump-diffusion dynamics.
    Extends the BasketPut model by including jump components in the simulation.
    """

    def __init__(self, eqn_config):
        super(JumpBasketPut, self).__init__(eqn_config)
        self.dim = eqn_config.dim
        self.total_time = eqn_config.total_time
        self.num_time_interval = eqn_config.num_time_interval
        self.delta_t = self.total_time / self.num_time_interval
        self.sqrt_delta_t = np.sqrt(self.delta_t)
        self.x_init = np.ones(self.dim) * 100.0
        self.sigma = 0.2
        self.r = 0.05
        self.K = 100.0

        self.mu = 0.02
        self.jump_size = 5.0
        self.jump_intensity = 0.1

    def sample(self, num_sample):
        return simulate_paths_with_jumps(
            X0=self.x_init,
            mu=self.mu,
            sigma=self.sigma,
            J=self.jump_size,
            lambda_=self.jump_intensity,
            N=self.num_time_interval,
            T=self.total_time,
            M=num_sample,
            d=self.dim,
        )

    def f_tf(self, t, x, y, z):
        return -self.r * y

    def g_tf(self, t, x):
        min_x = tf.reduce_min(x, axis=1, keepdims=True)
        return tf.maximum(self.K - min_x, 0)
