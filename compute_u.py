import numpy as np
import tensorflow as tf


def compute_u(model, bsde, x_input, t_input):
    """
    Compute u(x, t) using the trained BSDE model.

    Parameters:
    - model: Instance of NonsharedModel (trained).
    - bsde: Instance of the PDE class (e.g., HJBLQ).
    - x_input: Numpy array of shape [dim,].
    - t_input: Float representing the time t.

    Returns:
    - u_value: Predicted value of u(x, t).
    """
    # Validate inputs
    assert (
        x_input.shape[0] == bsde.dim
    ), "Dimension mismatch between x_input and bsde.dim"
    assert 0 <= t_input <= bsde.total_time, "t_input must be within [0, total_time]"

    # Determine the corresponding time step
    delta_t = bsde.delta_t
    time_step = int(t_input / delta_t)
    time_step = min(time_step, bsde.num_time_interval - 1)  # Ensure within bounds

    # Simulate a single path from t to T
    num_sample = 1
    # Generate random increments (dw) from t to T
    remaining_steps = bsde.num_time_interval - time_step
    dw = np.random.normal(
        loc=0.0, scale=np.sqrt(delta_t), size=[num_sample, bsde.dim, remaining_steps]
    )

    # Initialize x at time t
    x_current = np.expand_dims(x_input, axis=0)  # Shape: [1, dim]
    x_current = np.expand_dims(x_current, axis=2)  # Shape: [1, dim, 1]

    # Concatenate current x with future increments
    x_future = x_current
    for step in range(remaining_steps):
        x_next = x_future[:, :, -1] + bsde.sigma * dw[:, :, step]
        x_future = np.concatenate((x_future, np.expand_dims(x_next, axis=2)), axis=2)

    # Prepare inputs for the model
    dw_model_input = dw[:, :, :-1]  # Exclude the last increment for consistency
    x_model_input = x_future[:, :, 1:-1]  # Exclude initial and last state
    inputs = (dw_model_input, x_model_input)

    # Compute y and z using the model
    y_pred = model.call(inputs, training=False).numpy()[0, 0]

    return y_pred


from solver import BSDESolver

model = bsde_solver.model
bsde = bsde_solver.bsde

# Example 1: Single Point Evaluation
x_specific = np.ones(bsde.dim) * 100.0  # All state variables set to 100
t_specific = 0.5  # Time between 0 and T=1.0

u_value = compute_u(model, bsde, x_specific, t_specific)
print(f"u(x={x_specific}, t={t_specific}) = {u_value:.4f}")

# Example 2: Varying One Dimension
dim_to_vary = 0
x_values = np.linspace(80, 120, 50)
t_specific = 0.5

u_values = []
for x_val in x_values:
    x = np.ones(bsde.dim) * 100.0
    x[dim_to_vary] = x_val
    u = compute_u(model, bsde, x, t_specific)
    u_values.append(u)

plt.figure(figsize=(8, 6))
plt.plot(x_values, u_values, marker="o")
plt.title(f"Solution $u(x_{{{dim_to_vary+1}}}=x, t={t_specific})$")
plt.xlabel(f"$x_{dim_to_vary+1}$")
plt.ylabel("$u(x, t)$")
plt.grid(True)
plt.show()

# Example 3: Heatmap for Two Varying Dimensions
dim1, dim2 = 0, 1
x1_values = np.linspace(80, 120, 20)
x2_values = np.linspace(80, 120, 20)
X1, X2 = np.meshgrid(x1_values, x2_values)
u_grid = np.zeros_like(X1)

t_specific = 0.5

for i in range(X1.shape[0]):
    for j in range(X1.shape[1]):
        x = np.ones(bsde.dim) * 100.0
        x[dim1] = X1[i, j]
        x[dim2] = X2[i, j]
        u_grid[i, j] = compute_u(model, bsde, x, t_specific)

plt.figure(figsize=(10, 8))
cp = plt.contourf(X1, X2, u_grid, levels=50, cmap="viridis")
plt.colorbar(cp)
plt.title(f"Solution $u(x_{{{dim1+1}}}, x_{{{dim2+1}}}, t={t_specific})$")
plt.xlabel(f"$x_{dim1+1}$")
plt.ylabel(f"$x_{dim2+1}$")
plt.show()
