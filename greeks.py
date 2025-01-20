import tensorflow as tf


def compute_greeks(Z_network, x):
    """
    Computes Delta and Gamma for a given input state x using the provided Z_network.
    Z_network should approximate the gradient ∇ₓu(t, x).
    """
    x_tensor = tf.convert_to_tensor(x.reshape(1, -1), dtype=Z_network.dtype)

    with tf.GradientTape() as tape:
        tape.watch(x_tensor)
        Z = Z_network(x_tensor)
    delta = tape.gradient(Z, x_tensor).numpy().flatten()

    with tf.GradientTape() as tape2:
        tape2.watch(x_tensor)
        with tf.GradientTape() as tape1:
            tape1.watch(x_tensor)
            Z_inner = Z_network(x_tensor)
        first_deriv = tape1.gradient(Z_inner, x_tensor)
    gamma = tape2.gradient(first_deriv, x_tensor).numpy().flatten()

    return delta, gamma
