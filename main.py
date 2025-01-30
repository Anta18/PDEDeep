"""
The main file to run BSDE solver to solve parabolic partial differential equations (PDEs)
with integration of OHLC data fetched from Yahoo Finance using yfinance for quantitative finance applications.
"""

import json
import os
import logging
import pandas as pd
import numpy as np
import tensorflow as tf
import yfinance as yf  # Import for fetching OHLC data

from absl import app
from absl import flags
from absl import logging as absl_logging

import equation as eqn
from solver import BSDESolver
from greeks import compute_greeks

# Define command-line flags
flags.DEFINE_string(
    "config_path",
    "configs/option_pricing.json",
    """The path to load json configuration file.""",
)
flags.DEFINE_string(
    "exp_name",
    "european_call",
    """The name of the numerical experiment, prefix for logging.""",
)
flags.DEFINE_string(
    "ticker", "AAPL", """The ticker symbol of the asset to fetch OHLC data for."""
)
flags.DEFINE_string(
    "start_date", "2020-01-01", """The start date for fetching OHLC data."""
)
flags.DEFINE_string(
    "end_date", "2023-12-31", """The end date for fetching OHLC data."""
)
FLAGS = flags.FLAGS
FLAGS.log_dir = "./logs"


def main(argv):
    del argv  # Unused.

    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)

    # Clear previous TensorFlow sessions
    tf.keras.backend.clear_session()

    # Load configuration file
    with open(FLAGS.config_path) as json_data_file:
        config_dict = json.load(json_data_file)

    # Convert dictionary to object for attribute access
    class DictToObject:
        def __init__(self, dictionary):
            self._dict = dictionary
            for key, value in dictionary.items():
                setattr(self, key, value)

        def to_dict(self):
            return self._dict

    class Config:
        def __init__(self, config_dict):
            self.eqn_config = DictToObject(config_dict["eqn_config"])
            self.net_config = DictToObject(config_dict["net_config"])
            self._original_dict = config_dict

        def to_dict(self):
            return self._original_dict

    config = Config(config_dict)
    tf.keras.backend.set_floatx(config.net_config.dtype)

    # Fetch OHLC data using yfinance
    ticker = FLAGS.ticker
    start_date = FLAGS.start_date
    end_date = FLAGS.end_date

    logging.info(f"Fetching OHLC data for {ticker} from {start_date} to {end_date}.")

    try:
        ohlc_df = yf.download(ticker, start=start_date, end=end_date)
    except Exception as e:
        logging.error(f"Error fetching data for {ticker}: {e}")
        return

    if ohlc_df.empty:
        logging.error(
            f"No data fetched for ticker {ticker}. Please check the ticker symbol and date range."
        )
        return

    # Calculate log returns
    ohlc_df["Log_Return"] = np.log(ohlc_df["Close"] / ohlc_df["Close"].shift(1))
    ohlc_df.dropna(inplace=True)

    # Calculate rolling historical volatility (e.g., 30-day window, annualized)
    window_size = 30
    ohlc_df["Volatility"] = ohlc_df["Log_Return"].rolling(
        window=window_size
    ).std() * np.sqrt(252)

    # Handle cases where volatility might be NaN due to insufficient data points
    ohlc_df["Volatility"].fillna(method="bfill", inplace=True)

    # Estimate drift (mu) and volatility (sigma)
    mean_log_return = ohlc_df["Log_Return"].mean()
    historical_volatility = ohlc_df["Volatility"].mean()

    # Define risk-free rate (e.g., 2%)
    risk_free_rate = 0.02

    # Drift under risk-neutral measure is typically set to risk-free rate
    mu = risk_free_rate

    # Volatility from historical data
    sigma = historical_volatility

    logging.info(f"Estimated drift (mu): {mu}")
    logging.info(f"Estimated volatility (sigma): {sigma}")

    # Instantiate the PDE equation class with financial parameters
    # Assuming 'PricingDefaultRisk' is the target equation for option pricing
    try:
        bsde = getattr(eqn, config.eqn_config.eqn_name)(
            config.eqn_config, mu=mu, sigma=sigma, rate=risk_free_rate
        )
    except AttributeError:
        logging.error(
            f"Equation class '{config.eqn_config.eqn_name}' not found in equation.py."
        )
        return

    # Initialize BSDE solver with the equation and configuration
    bsde_solver = BSDESolver(config, bsde, log_dir=FLAGS.log_dir)

    # Ensure the log directory exists
    os.makedirs(FLAGS.log_dir, exist_ok=True)
    path_prefix = os.path.join(FLAGS.log_dir, FLAGS.exp_name)

    # Save the current configuration for reference
    with open(f"{path_prefix}_config.json", "w") as outfile:
        json.dump(config.to_dict(), outfile, indent=2)

    # Configure logging format and verbosity
    absl_logging.get_absl_handler().setFormatter(
        logging.Formatter("%(levelname)-6s %(message)s")
    )
    absl_logging.set_verbosity("info")

    logging.info(f"Begin to solve {config.eqn_config.eqn_name} for {ticker}.")

    # Train the model
    training_history = bsde_solver.train()

    # Log the true initial condition if available
    if bsde.y_init:
        logging.info(f"Y0_true: {bsde.y_init:.4e}")
        relative_error = abs(bsde.y_init - training_history[-1, 2]) / bsde.y_init
        logging.info(f"Relative error of Y0: {relative_error:.2%}")

    # Save training history to CSV for analysis
    np.savetxt(
        f"{path_prefix}_training_history.csv",
        training_history,
        fmt=["%d", "%.5e", "%.5e", "%d"],
        delimiter=",",
        header="step,loss_function,target_value,elapsed_time",
        comments="",
    )

    # Compute Greeks (Delta and Gamma) using the trained model
    model = bsde_solver.model
    subnet_for_greeks = model.subnet[0]
    dim = config.eqn_config.dim

    # Example input asset price (can be adjusted as needed)
    x_example = np.full((dim,), 100.0)

    # Compute Greeks
    delta, gamma = compute_greeks(subnet_for_greeks, x_example)
    logging.info(f"Computed Delta (using subnet[0]): {delta}")
    logging.info(f"Computed Gamma (using subnet[0]): {gamma}")


if __name__ == "__main__":
    app.run(main)
