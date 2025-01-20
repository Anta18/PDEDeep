this is my code right now for the research paper improve it further:

main.py
"""
The main file to run BSDE solver to solve parabolic partial differential equations (PDEs).

"""

import json
import os
import logging

from absl import app
from absl import flags
from absl import logging as absl_logging
import numpy as np
import tensorflow as tf

import equation as eqn
from solver import BSDESolver

flags.DEFINE_string('config_path', 'configs/hjb_lq_d100.json',
"""The path to load json file.""")
flags.DEFINE_string('exp_name', 'test',
"""The name of numerical experiments, prefix for logging""")
FLAGS = flags.FLAGS
FLAGS.log_dir = './logs' # directory where to write event logs and output array

def main(argv):
del argv
tf.keras.backend.clear_session()
with open(FLAGS.config_path) as json_data_file:
config_dict = json.load(json_data_file)

    class DictToObject:
        def __init__(self, dictionary):
            self._dict = dictionary
            for key, value in dictionary.items():
                setattr(self, key, value)

        def to_dict(self):
            return self._dict

    class Config:
        def __init__(self, config_dict):
            self.eqn_config = DictToObject(config_dict['eqn_config'])
            self.net_config = DictToObject(config_dict['net_config'])
            self._original_dict = config_dict

        def to_dict(self):
            return self._original_dict

    config = Config(config_dict)
    tf.keras.backend.set_floatx(config.net_config.dtype)

    bsde = getattr(eqn, config.eqn_config.eqn_name)(config.eqn_config)
    bsde_solver = BSDESolver(config, bsde)

    if not os.path.exists(FLAGS.log_dir):
        os.mkdir(FLAGS.log_dir)
    path_prefix = os.path.join(FLAGS.log_dir, FLAGS.exp_name)
    with open('{}_config.json'.format(path_prefix), 'w') as outfile:
        json.dump(config.to_dict(), outfile, indent=2)

    absl_logging.get_absl_handler().setFormatter(logging.Formatter('%(levelname)-6s %(message)s'))
    absl_logging.set_verbosity('info')

    logging.info('Begin to solve %s ' % config.eqn_config.eqn_name)
    training_history = bsde_solver.train()
    if bsde.y_init:
        logging.info('Y0_true: %.4e' % bsde.y_init)
        logging.info('relative error of Y0: %s',
                     '{:.2%}'.format(abs(bsde.y_init - training_history[-1, 2])/bsde.y_init))
    np.savetxt('{}_training_history.csv'.format(path_prefix),
               training_history,
               fmt=['%d', '%.5e', '%.5e', '%d'],
               delimiter=",",
               header='step,loss_function,target_value,elapsed_time',
               comments='')

if **name** == '**main**':

    app.run(main)

equation.py

import numpy as np
import tensorflow as tf

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
PDE: u_t + (1/2)\*Δu = 0 with a given terminal condition u(T,x) = g(x).
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
u*t + 0.5 \* sum*{i=1}^d sigma^2 x*i^2 u*{x*i x_i} + r sum*{i=1}^d x*i u*{x_i} - r u = 0,
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
u*t + 0.5 \* sum*{i=1}^d sigma^2 x*i^2 u*{x*i x_i} + r sum*{i=1}^d x*i u*{x_i} - r u = 0,
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
u*t + 0.5 \* sum*{i=1}^d sigma^2 x*i^2 u*{x*i x_i} + r sum*{i=1}^d x*i u*{x_i} - r u = 0,
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

solver.py

import logging
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

DELTA_CLIP = 50.0

class BSDESolver(object):
def **init**(self, config, bsde):
self.eqn_config = config.eqn_config
self.net_config = config.net_config
self.bsde = bsde

        self.model = NonsharedModel(config, bsde)
        lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
            self.net_config.lr_boundaries, self.net_config.lr_values
        )
        self.optimizer = tf.keras.optimizers.Adam(
            learning_rate=lr_schedule, epsilon=1e-8
        )

    def train(self):
        start_time = time.time()
        training_history = []
        valid_data = self.bsde.sample(self.net_config.valid_size)
        # begin sgd iteration
        for step in range(self.net_config.num_iterations + 1):
            if step % self.net_config.logging_frequency == 0:
                loss = self.loss_fn(valid_data, training=False).numpy()
                y_init = self.model.y_init.numpy()[0]
                elapsed_time = time.time() - start_time
                training_history.append([step, loss, y_init, elapsed_time])

                if self.net_config.verbose:
                    logging.info(
                        "step: %5u,    loss: %.4e, Y0: %.4e,   elapsed time: %3u"
                        % (step, loss, y_init, elapsed_time)
                    )
            self.train_step(self.bsde.sample(self.net_config.batch_size))

        return np.array(training_history)

    def loss_fn(self, inputs, training):
        dw, x = inputs
        y_terminal = self.model(inputs, training=training)
        delta = y_terminal - self.bsde.g_tf(self.bsde.total_time, x[:, :, -1])
        # use linear approximation outside the clipped range
        loss = tf.reduce_mean(
            tf.where(
                tf.abs(delta) < DELTA_CLIP,
                tf.square(delta),
                2 * DELTA_CLIP * tf.abs(delta) - DELTA_CLIP**2,
            )
        )

        return loss

    def grad(self, inputs, training):
        with tf.GradientTape(persistent=True) as tape:
            loss = self.loss_fn(inputs, training)
        grad = tape.gradient(loss, self.model.trainable_variables)
        del tape
        return grad

    @tf.function
    def train_step(self, train_data):
        with tf.GradientTape() as tape:
            loss = self.loss_fn(train_data, training=True)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

class NonsharedModel(tf.keras.Model):
def **init**(self, config, bsde):
super().**init**()
self.eqn_config = config.eqn_config
self.net_config = config.net_config
self.bsde = bsde
y_init_range = self.net_config.y_init_range

        self.y_init = self.add_weight(
            name="y_init",
            shape=[1],
            initializer=tf.random_uniform_initializer(y_init_range[0], y_init_range[1]),
        )
        self.z_init = self.add_weight(
            name="z_init",
            shape=[1, self.eqn_config.dim],
            initializer=tf.random_uniform_initializer(-0.1, 0.1),
        )

        self.subnet = [
            FeedForwardSubNet(config) for _ in range(self.bsde.num_time_interval - 1)
        ]

    def call(self, inputs, training=None):
        dw, x = inputs
        time_stamp = np.arange(0, self.eqn_config.num_time_interval) * self.bsde.delta_t
        all_one_vec = tf.ones(
            shape=tf.stack([tf.shape(dw)[0], 1]), dtype=self.net_config.dtype
        )
        y = all_one_vec * self.y_init
        z = tf.matmul(all_one_vec, self.z_init)

        for t in range(0, self.bsde.num_time_interval - 1):
            y = (
                y
                - self.bsde.delta_t * (self.bsde.f_tf(time_stamp[t], x[:, :, t], y, z))
                + tf.reduce_sum(z * dw[:, :, t], 1, keepdims=True)
            )
            z = self.subnet[t](x[:, :, t + 1], training=training) / self.bsde.dim

        # terminal time
        y = (
            y
            - self.bsde.delta_t * self.bsde.f_tf(time_stamp[-1], x[:, :, -2], y, z)
            + tf.reduce_sum(z * dw[:, :, -1], 1, keepdims=True)
        )

        return y

class FeedForwardSubNet(tf.keras.Model):
def **init**(self, config):
super(FeedForwardSubNet, self).**init**()
dim = config.eqn_config.dim
num_hiddens = config.net_config.num_hiddens

        self.bn_layers = [
            tf.keras.layers.BatchNormalization(
                momentum=0.99,
                epsilon=1e-6,
                beta_initializer=tf.random_normal_initializer(0.0, stddev=0.1),
                gamma_initializer=tf.random_uniform_initializer(0.1, 0.5),
            )
            for _ in range(len(num_hiddens) + 2)
        ]
        self.dense_layers = [
            tf.keras.layers.Dense(num_hiddens[i], use_bias=False, activation=None)
            for i in range(len(num_hiddens))
        ]
        # final output should be gradient of size dim
        self.dense_layers.append(tf.keras.layers.Dense(dim, activation=None))

    def call(self, x, training=None):
        """structure: bn -> (dense -> bn -> relu) * len(num_hiddens) -> dense -> bn"""
        x = self.bn_layers[0](x, training=training)
        for i in range(len(self.dense_layers) - 1):
            x = self.dense_layers[i](x)
            x = self.bn_layers[i + 1](x, training=training)
            x = tf.nn.relu(x)
        x = self.dense_layers[-1](x)
        x = self.bn_layers[-1](x, training=training)

        return x

def create_Z_network_with_domain(input_dim, num_domains, layers_sizes=[100, 100, 100]):
"""
Creates a neural network that takes both asset features and domain indicators.
input_dim: original dimension of asset features.
num_domains: number of possible domains (market regimes).
"""
total_input_dim = input_dim + num_domains
inputs = layers.Input(shape=(total_input_dim,))
x = inputs
for size in layers_sizes:
x = layers.Dense(size, activation="relu")(x)
x = layers.BatchNormalization()(x)
outputs = layers.Dense(input_dim)(
x
) # Output dimension matches asset dimension for Z_t
model = models.Model(inputs=inputs, outputs=outputs)
return model

simulation.py
def simulate*paths_with_jumps(X0, mu, sigma, J, lambda*, N, T, M, d):
"""
Simulate M paths of a d-dimensional jump-diffusion process over N steps with: - X0: initial asset prices, - mu: drift vector, - sigma: volatility vector, - J: jump size (can be a vector or scalar), - lambda*: intensity of the Poisson process.
"""
dt = T / N
paths = np.zeros((M, N + 1, d))
paths[:, 0, :] = X0
for n in range(1, N + 1):
Z = np.random.normal(size=(M, d)) # Simulate Poisson jumps for each asset
jumps = np.random.poisson(lambda* _ dt, size=(M, d))
jump_component = J _ jumps
paths[:, n, :] = (
paths[:, n - 1, :] + mu _ paths[:, n - 1, :] _ dt + sigma _ paths[:, n - 1, :] _ np.sqrt(dt) \* Z + jump_component
)
return paths

greeks.py

import tensorflow as tf

def compute_greeks(Z_network, x):
"""
Computes Delta and Gamma for a given input state x using the provided Z_network.
Z_network should approximate the gradient ∇ₓu(t, x).
"""
x_tensor = tf.convert_to_tensor(x.reshape(1, -1), dtype=tf.float64)

    # Compute Delta (first derivative)
    with tf.GradientTape() as tape:
        tape.watch(x_tensor)
        Z = Z_network(x_tensor)
    delta = tape.gradient(Z, x_tensor).numpy().flatten()

    # Compute Gamma (second derivative) using nested GradientTapes
    with tf.GradientTape() as tape2:
        tape2.watch(x_tensor)
        with tf.GradientTape() as tape1:
            tape1.watch(x_tensor)
            Z_inner = Z_network(x_tensor)
        first_deriv = tape1.gradient(Z_inner, x_tensor)
    gamma = tape2.gradient(first_deriv, x_tensor).numpy().flatten()

    return delta, gamma

config files:

configs/allencahn_d100.json
{
"eqn_config": {
"\_comment": "Allen-Cahn equation in PNAS paper doi.org/10.1073/pnas.1718942115",
"eqn_name": "AllenCahn",
"total_time": 0.3,
"dim": 100,
"num_time_interval": 20
},
"net_config": {
"y_init_range": [0.3, 0.5],
"num_hiddens": [110, 110],
"lr_values": [5e-4, 5e-4],
"lr_boundaries": [2000],
"num_iterations": 4000,
"batch_size": 64,
"valid_size": 256,
"logging_frequency": 100,
"dtype": "float64",
"verbose": true
}
}

configs/burgers_type_d50.json

{
"eqn_config": {
"\_comment": "Multidimensional Burgers-type PDE in Comm. Math. Stat. doi.org/10.1007/s40304-017-0117-6",
"eqn_name": "BurgersType",
"total_time": 0.2,
"dim": 50,
"num_time_interval": 30
},
"net_config": {
"y_init_range": [2, 4],
"num_hiddens": [60, 60],
"lr_values": [1e-2, 1e-3, 1e-4],
"lr_boundaries": [15000, 25000],
"num_iterations": 30000,
"batch_size": 64,
"valid_size": 256,
"logging_frequency": 100,
"dtype": "float64",
"verbose": true
}
}

configs/hjb_lq_d100.json

{
"eqn_config": {
"\_comment": "HJB equation in PNAS paper doi.org/10.1073/pnas.1718942115",
"eqn_name": "HJBLQ",
"total_time": 1.0,
"dim": 100,
"num_time_interval": 20
},
"net_config": {
"y_init_range": [0, 1],
"num_hiddens": [110, 110],
"lr_values": [1e-2, 1e-2],
"lr_boundaries": [1000],
"num_iterations": 2000,
"batch_size": 64,
"valid_size": 256,
"logging_frequency": 100,
"dtype": "float64",
"verbose": true
}
}

configs/pricing_default_risk_d100.json

{
"eqn_config": {
"\_comment": "Nonlinear Black-Scholes equation with default risk in PNAS paper doi.org/10.1073/pnas.1718942115",
"eqn_name": "PricingDefaultRisk",
"total_time": 1.0,
"dim": 100,
"num_time_interval": 40
},
"net_config": {
"y_init_range": [40, 50],
"num_hiddens": [110, 110],
"lr_values": [8e-3, 8e-3],
"lr_boundaries": [3000],
"num_iterations": 6000,
"batch_size": 64,
"valid_size": 256,
"logging_frequency": 100,
"dtype": "float64",
"verbose": true
}
}

configs/pricing_diffrate_d100.json
{
"eqn_config": {
"\_comment": "Nonlinear Black-Scholes equation with different interest rates for borrowing and lending in Comm. Math. Stat. doi.org/10.1007/s40304-017-0117-6",
"eqn_name": "PricingDiffRate",
"total_time": 0.5,
"dim": 100,
"num_time_interval": 20
},
"net_config": {
"y_init_range": [15, 18],
"num_hiddens": [110, 110],
"lr_values": [5e-3, 5e-3],
"lr_boundaries": [2000],
"num_iterations": 4000,
"batch_size": 64,
"valid_size": 256,
"logging_frequency": 100,
"dtype": "float64",
"verbose": true
}
}

configs/quad_grad_d100.json

{
"eqn_config": {
"\_comment": "An example PDE with quadratically growing derivatives in Comm. Math. Stat. doi.org/10.1007/s40304-017-0117-6",
"eqn_name": "QuadraticGradient",
"total_time": 1.0,
"dim": 100,
"num_time_interval": 30
},
"net_config": {
"y_init_range": [2, 4],
"num_hiddens": [110, 110],
"lr_values": [5e-3, 5e-3],
"lr_boundaries": [2000],
"num_iterations": 4000,
"batch_size": 64,
"valid_size": 256,
"logging_frequency": 100,
"dtype": "float64",
"verbose": true
}
}

configs/reaction_diffusion_d100.json

{
"eqn_config": {
"\_comment": "Time-dependent reaction-diffusion-type example PDE in Comm. Math. Stat. doi.org/10.1007/s40304-017-0117-6",
"eqn_name": "ReactionDiffusion",
"total_time": 1.0,
"dim": 100,
"num_time_interval": 30
},
"net_config": {
"y_init_range": [0, 1],
"num_hiddens": [110, 110],
"lr_values": [1e-2, 1e-2],
"lr_boundaries": [120],
"num_iterations": 240,
"batch_size": 64,
"valid_size": 256,
"logging_frequency": 100,
"dtype": "float64",
"verbose": true
}
}

basket_option_d100.json

{
"eqn_config": {
"\_comment": "European Basket Call Option PDE pricing",
"eqn_name": "BasketOption",
"total_time": 1.0,
"dim": 100,
"num_time_interval": 20
},
"net_config": {
"y_init_range": [0, 1],
"num_hiddens": [110, 110],
"lr_values": [1e-2, 1e-2],
"lr_boundaries": [1000],
"num_iterations": 2000,
"batch_size": 64,
"valid_size": 256,
"logging_frequency": 100,
"dtype": "float64",
"verbose": true
}
}

heat_equation_d100.json
{
"eqn_config": {
"\_comment": "Simple Heat Equation",
"eqn_name": "HeatEquation",
"total_time": 1.0,
"dim": 100,
"num_time_interval": 20
},
"net_config": {
"y_init_range": [0.0, 0.1],
"num_hiddens": [110, 110],
"lr_values": [1e-2, 1e-2],
"lr_boundaries": [1000],
"num_iterations": 2000,
"batch_size": 64,
"valid_size": 256,
"logging_frequency": 100,
"dtype": "float64",
"verbose": true
}
}

readme:

# [Deep BSDE Solver](https://doi.org/10.1073/pnas.1718942115) in TensorFlow

## Quick Installation

For a quick installation, you can create a conda environment for Python using the following command:

bash
conda env create -f environment.yml

## Training

python main.py --config_path=configs/hjb_lq_d100.json

Command-line flags:

- config_path: Config path corresponding to the partial differential equation (PDE) to solve.
  There are seven PDEs implemented so far. See [Problems](#problems) section below.
- exp_name: Name of numerical experiment, prefix of logging and output.
- log_dir: Directory to write logging and output array.

## Problems

equation.py and config.py now support the following problems:

Three examples in ref [1]:

- HJBLQ: Hamilton-Jacobi-Bellman (HJB) equation.
- AllenCahn: Allen-Cahn equation with a cubic nonlinearity.
- PricingDefaultRisk: Nonlinear Black-Scholes equation with default risk in consideration.

Four examples in ref [2]:

- PricingDiffRate: Nonlinear Black-Scholes equation for the pricing of European financial derivatives
  with different interest rates for borrowing and lending.
- BurgersType: Multidimensional Burgers-type PDEs with explicit solution.
- QuadraticGradient: An example PDE with quadratically growing derivatives and an explicit solution.
- ReactionDiffusion: Time-dependent reaction-diffusion-type example PDE with oscillating explicit solutions.

New problems can be added very easily. Inherit the class equation
in equation.py and define the new problem. Note that the generator function
and terminal function should be TensorFlow operations while the sample function
can be python operation. A proper config is needed as well.

## Dependencies

The code is compatible with TensorFlow >=2.16 and Keras 3.

For those using older versions

- a version of the Deep BSDE solver that is compatible with TensorFlow 2.13 can be found in commit [5af8c25](https://github.com/frankhan91/DeepBSDE/commit/5af8c25).

- \*a version of the Deep BSDE solver that is compatible with TensorFlow 1.12 and Python 2 can be found in commit [9d4e332](https://github.com/frankhan91/DeepBSDE/commit/9d4e3329613d7a2a1f389a0d4cc652e1c4606b86).

## Reference

[1] Han, J., Jentzen, A., and E, W. Overcoming the curse of dimensionality: Solving high-dimensional partial differential equations using deep learning,
<em>Proceedings of the National Academy of Sciences</em>, 115(34), 8505-8510 (2018). [[journal]](https://doi.org/10.1073/pnas.1718942115) [[arXiv]](https://arxiv.org/abs/1707.02568) <br />
[2] E, W., Han, J., and Jentzen, A. Deep learning-based numerical methods for high-dimensional parabolic partial differential equations and backward stochastic differential equations,
<em>Communications in Mathematics and Statistics</em>, 5, 349–380 (2017).
[[journal]](https://doi.org/10.1007/s40304-017-0117-6) [[arXiv]](https://arxiv.org/abs/1706.04702)
