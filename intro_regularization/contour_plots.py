import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import os
import matplotlib
from mpl_toolkits.mplot3d import Axes3D

matplotlib.use('TkAgg')

# Ensure output directory exists
output_dir = 'regularization_plots'
os.makedirs(output_dir, exist_ok=True)


def create_error_grid(beta_hat):
    """Create a grid for error calculations."""
    x = np.linspace(-2, 2, 100)
    y = np.linspace(-2, 2, 100)
    X, Y = np.meshgrid(x, y)

    def least_squares_error(x, y):
        return (x - beta_hat[0]) ** 2 + (y - beta_hat[1]) ** 2

    Z = least_squares_error(X, Y)
    return X, Y, Z


def plot_ridge_regularization(beta_hat):
    plt.figure(figsize=(10, 8))
    X, Y, Z = create_error_grid(beta_hat)

    plt.contourf(X, Y, Z, levels=20, cmap='viridis', alpha=0.7)
    plt.colorbar(label='Error')

    # Draw L2 constraint (circle)
    theta = np.linspace(0, 2 * np.pi, 100)
    radius = 1
    plt.plot(radius * np.cos(theta), radius * np.sin(theta), 'r-', linewidth=2)

    # Compute error values
    def least_squares_error(x, y):
        return (x - beta_hat[0]) ** 2 + (y - beta_hat[1]) ** 2

    # Original least squares point (red)
    ls_error = least_squares_error(beta_hat[0], beta_hat[1])
    plt.plot(beta_hat[0], beta_hat[1], 'r*', markersize=15)
    plt.annotate(f'Least Squares Solution\nError: {ls_error:.4f}',
                 (beta_hat[0], beta_hat[1]),
                 xytext=(10, 10),
                 textcoords='offset points',
                 color='red')

    # Ridge solution (exactly on the border)
    def ridge_objective(beta):
        return np.sum((beta - beta_hat) ** 2)

    def ridge_constraint(beta):
        return radius ** 2 - np.sum(beta ** 2)

    ridge_solution = minimize(
        ridge_objective,
        beta_hat,
        constraints={'type': 'ineq', 'fun': ridge_constraint}
    ).x

    # Compute error for ridge solution
    ridge_error = least_squares_error(ridge_solution[0], ridge_solution[1])

    plt.plot(ridge_solution[0], ridge_solution[1], 'go', markersize=10)
    plt.annotate(
        f'Ridge-Regularized Solution\nError: {ridge_error:.4f}\n(β1: {ridge_solution[0]:.4f}, β2: {ridge_solution[1]:.4f})',
        (ridge_solution[0], ridge_solution[1]),
        xytext=(10, 10),
        textcoords='offset points',
        color='green')

    plt.title('Ridge Regularization (L2 Constraint)')
    plt.xlabel('β1')
    plt.ylabel('β2')
    plt.axhline(y=0, color='k', linewidth=0.5)
    plt.axvline(x=0, color='k', linewidth=0.5)
    plt.grid(True, linestyle='--', linewidth=0.5)

    plt.savefig(os.path.join(output_dir, 'ridge_regularization.png'))
    plt.close()


def plot_lasso_regularization(beta_hat):
    plt.figure(figsize=(10, 8))
    X, Y, Z = create_error_grid(beta_hat)

    plt.contourf(X, Y, Z, levels=20, cmap='viridis', alpha=0.7)
    plt.colorbar(label='Error')

    # Draw L1 constraint (diamond)
    diamond_x = [0, 1, 0, -1, 0]
    diamond_y = [1, 0, -1, 0, 1]
    plt.plot(diamond_x, diamond_y, 'r-', linewidth=2)

    # Compute error values
    def least_squares_error(x, y):
        return (x - beta_hat[0]) ** 2 + (y - beta_hat[1]) ** 2

    # Original least squares point (red)
    ls_error = least_squares_error(beta_hat[0], beta_hat[1])
    plt.plot(beta_hat[0], beta_hat[1], 'r*', markersize=15)
    plt.annotate(f'Least Squares Solution\nError: {ls_error:.4f}',
                 (beta_hat[0], beta_hat[1]),
                 xytext=(10, 10),
                 textcoords='offset points',
                 color='red')

    # Lasso solution (exactly on the border)
    def lasso_objective(beta):
        return np.sum((beta - beta_hat) ** 2)

    def lasso_constraint(beta):
        return 1 - np.sum(np.abs(beta))

    lasso_solution = minimize(
        lasso_objective,
        beta_hat,
        constraints={'type': 'ineq', 'fun': lasso_constraint}
    ).x

    # Compute error for lasso solution
    lasso_error = least_squares_error(lasso_solution[0], lasso_solution[1])

    plt.plot(lasso_solution[0], lasso_solution[1], 'go', markersize=10)
    plt.annotate(
        f'Lasso-Regularized Solution\nError: {lasso_error:.4f}\n(β1: {lasso_solution[0]:.4f}, β2: {lasso_solution[1]:.4f})',
        (lasso_solution[0], lasso_solution[1]),
        xytext=(10, 10),
        textcoords='offset points',
        color='green')

    plt.title('Lasso Regularization (L1 Constraint)')
    plt.xlabel('β1')
    plt.ylabel('β2')
    plt.axhline(y=0, color='k', linewidth=0.5)
    plt.axvline(x=0, color='k', linewidth=0.5)
    plt.grid(True, linestyle='--', linewidth=0.5)

    plt.savefig(os.path.join(output_dir, 'lasso_regularization.png'))
    plt.close()


def plot_ridge_3d_error_surface(beta_hat):
    plt.figure(figsize=(12, 10))
    X, Y, Z = create_error_grid(beta_hat)

    ax = plt.axes(projection='3d')
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.7)
    plt.colorbar(surf, shrink=0.8, aspect=10)

    def least_squares_error(x, y):
        return (x - beta_hat[0]) ** 2 + (y - beta_hat[1]) ** 2

    def ridge_objective(beta):
        return np.sum((beta - beta_hat) ** 2)

    def ridge_constraint(beta):
        return 1 - np.sum(beta ** 2)

    # Original least squares point
    ls_error = least_squares_error(beta_hat[0], beta_hat[1])
    ax.scatter(beta_hat[0], beta_hat[1], ls_error,
               color='red', s=100, marker='*',
               label=f'Least Squares Solution\nError: {ls_error:.4f}')

    ridge_solution = minimize(
        ridge_objective,
        beta_hat,
        constraints={'type': 'ineq', 'fun': ridge_constraint}
    ).x

    # Ridge solution point
    ridge_error = least_squares_error(ridge_solution[0], ridge_solution[1])
    ax.scatter(ridge_solution[0], ridge_solution[1], ridge_error,
               color='green', s=100,
               label=f'Ridge Solution\nError: {ridge_error:.4f}\n(β1: {ridge_solution[0]:.4f}, β2: {ridge_solution[1]:.4f})')

    ax.set_title('3D Ridge Regularization Error Surface')
    ax.set_xlabel('β1')
    ax.set_ylabel('β2')
    ax.set_zlabel('Error')
    ax.legend()

    plt.savefig(os.path.join(output_dir, 'ridge_3d_error_surface.png'))
    plt.close()


def plot_lasso_3d_error_surface(beta_hat):
    plt.figure(figsize=(12, 10))
    X, Y, Z = create_error_grid(beta_hat)

    ax = plt.axes(projection='3d')
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.7)
    plt.colorbar(surf, shrink=0.8, aspect=10)

    def least_squares_error(x, y):
        return (x - beta_hat[0]) ** 2 + (y - beta_hat[1]) ** 2

    def lasso_objective(beta):
        return np.sum((beta - beta_hat) ** 2)

    def lasso_constraint(beta):
        return 1 - np.sum(np.abs(beta))

    # Original least squares point
    ls_error = least_squares_error(beta_hat[0], beta_hat[1])
    ax.scatter(beta_hat[0], beta_hat[1], ls_error,
               color='red', s=100, marker='*',
               label=f'Least Squares Solution\nError: {ls_error:.4f}')

    lasso_solution = minimize(
        lasso_objective,
        beta_hat,
        constraints={'type': 'ineq', 'fun': lasso_constraint}
    ).x

    # Lasso solution point
    lasso_error = least_squares_error(lasso_solution[0], lasso_solution[1])
    ax.scatter(lasso_solution[0], lasso_solution[1], lasso_error,
               color='green', s=100,
               label=f'Lasso Solution\nError: {lasso_error:.4f}\n(β1: {lasso_solution[0]:.4f}, β2: {lasso_solution[1]:.4f})')

    ax.set_title('3D Lasso Regularization Error Surface')
    ax.set_xlabel('β1')
    ax.set_ylabel('β2')
    ax.set_zlabel('Error')
    ax.legend()

    plt.savefig(os.path.join(output_dir, 'lasso_3d_error_surface.png'))
    plt.close()


# Other functions remain the same as in the previous script...

# Define the shifted optimal point (least squares solution)
beta_hat = np.array([1.5, 1.0])

# Generate all plots
plot_ridge_regularization(beta_hat)
plot_lasso_regularization(beta_hat)
plot_ridge_3d_error_surface(beta_hat)
plot_lasso_3d_error_surface(beta_hat)

print(f"Plots have been saved in the '{output_dir}' directory.")