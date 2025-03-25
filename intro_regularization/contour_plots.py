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

def plot_ridge_3d_error_surface(beta_hat):
    # Create a figure
    plt.figure(figsize=(12, 10))

    # Create grid of points centered around 0
    x = np.linspace(-2, 2, 100)
    y = np.linspace(-2, 2, 100)
    X, Y = np.meshgrid(x, y)

    # Least Squares Error Function
    def least_squares_error(x, y):
        return (x - beta_hat[0]) ** 2 + (y - beta_hat[1]) ** 2

    # Compute error
    Z = least_squares_error(X, Y)

    # Create 3D surface plot
    ax = plt.axes(projection='3d')
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.7)
    plt.colorbar(surf, shrink=0.8, aspect=10)

    # Draw L2 constraint (circle)
    theta = np.linspace(0, 2 * np.pi, 100)
    radius = 1

    # Ridge solution (exactly on the border)
    def ridge_objective(beta):
        return np.sum((beta - beta_hat) ** 2)

    def ridge_constraint(beta):
        return radius ** 2 - np.sum(beta ** 2)

    # Solve constrained optimization
    ridge_solution = minimize(
        ridge_objective,
        beta_hat,
        constraints={'type': 'ineq', 'fun': ridge_constraint}
    ).x

    # Get the error value for the ridge solution
    ridge_error = least_squares_error(ridge_solution[0], ridge_solution[1])

    # Plot the original least squares point (red)
    ax.scatter(beta_hat[0], beta_hat[1],
               least_squares_error(beta_hat[0], beta_hat[1]),
               color='red', s=100, marker='*',
               label='Least Squares Solution')

    # Plot the ridge solution point (green)
    ax.scatter(ridge_solution[0], ridge_solution[1], ridge_error,
               color='green', s=100,
               label=f'Ridge Solution\n(β1: {ridge_solution[0]:.4f}, β2: {ridge_solution[1]:.4f})')

    # Add annotations
    ax.text(ridge_solution[0], ridge_solution[1], ridge_error,
            f'  Ridge Solution\n  β1: {ridge_solution[0]:.4f}\n  β2: {ridge_solution[1]:.4f}',
            color='green')

    # Styling
    ax.set_title('3D Ridge Regularization Error Surface')
    ax.set_xlabel('β1')
    ax.set_ylabel('β2')
    ax.set_zlabel('Error')
    ax.legend()

    # Save the plot
    plt.savefig(os.path.join(output_dir, 'ridge_3d_error_surface.png'))
    plt.close()

def plot_lasso_3d_error_surface(beta_hat):
    # Create a figure
    plt.figure(figsize=(12, 10))

    # Create grid of points centered around 0
    x = np.linspace(-2, 2, 100)
    y = np.linspace(-2, 2, 100)
    X, Y = np.meshgrid(x, y)

    # Least Squares Error Function
    def least_squares_error(x, y):
        return (x - beta_hat[0]) ** 2 + (y - beta_hat[1]) ** 2

    # Compute error
    Z = least_squares_error(X, Y)

    # Create 3D surface plot
    ax = plt.axes(projection='3d')
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.7)
    plt.colorbar(surf, shrink=0.8, aspect=10)

    # Lasso solution (exactly on the border)
    def lasso_objective(beta):
        return np.sum((beta - beta_hat) ** 2)

    def lasso_constraint(beta):
        return 1 - np.sum(np.abs(beta))

    # Solve constrained optimization
    lasso_solution = minimize(
        lasso_objective,
        beta_hat,
        constraints={'type': 'ineq', 'fun': lasso_constraint}
    ).x

    # Get the error value for the lasso solution
    lasso_error = least_squares_error(lasso_solution[0], lasso_solution[1])

    # Plot the original least squares point (red)
    ax.scatter(beta_hat[0], beta_hat[1],
               least_squares_error(beta_hat[0], beta_hat[1]),
               color='red', s=100, marker='*',
               label='Least Squares Solution')

    # Plot the lasso solution point (green)
    ax.scatter(lasso_solution[0], lasso_solution[1], lasso_error,
               color='green', s=100,
               label=f'Lasso Solution\n(β1: {lasso_solution[0]:.4f}, β2: {lasso_solution[1]:.4f})')

    # Add annotations
    ax.text(lasso_solution[0], lasso_solution[1], lasso_error,
            f'  Lasso Solution\n  β1: {lasso_solution[0]:.4f}\n  β2: {lasso_solution[1]:.4f}',
            color='green')

    # Styling
    ax.set_title('3D Lasso Regularization Error Surface')
    ax.set_xlabel('β1')
    ax.set_ylabel('β2')
    ax.set_zlabel('Error')
    ax.legend()

    # Save the plot
    plt.savefig(os.path.join(output_dir, 'lasso_3d_error_surface.png'))
    plt.close()

# Define the shifted optimal point (least squares solution)
beta_hat = np.array([1.5, 1.0])

# Generate additional 3D plots
plot_ridge_3d_error_surface(beta_hat)
plot_lasso_3d_error_surface(beta_hat)

print(f"3D plots have been saved in the '{output_dir}' directory.")