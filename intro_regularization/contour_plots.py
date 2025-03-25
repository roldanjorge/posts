import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import matplotlib
matplotlib.use('TkAgg')


def create_comprehensive_regularization_plot():
    # Create a figure with multiple subplots
    fig = plt.figure(figsize=(16, 12))

    # Define the shifted optimal point (least squares solution)
    beta_hat = np.array([1.5, 1.0])

    # Create grid of points centered around 0
    x = np.linspace(-2, 2, 100)
    y = np.linspace(-2, 2, 100)
    X, Y = np.meshgrid(x, y)

    # Least Squares Error Function
    def least_squares_error(x, y):
        return (x - beta_hat[0]) ** 2 + (y - beta_hat[1]) ** 2

    # Function to set up common plot features
    def setup_plot(ax, title):
        ax.set_xlabel('β1')
        ax.set_ylabel('β2')
        ax.set_title(title)
        ax.axhline(y=0, color='k', linewidth=0.5)
        ax.axvline(x=0, color='k', linewidth=0.5)
        ax.grid(True, linestyle='--', linewidth=0.5)

    # 2D Visualization
    # Least Squares Error Contours
    ax1 = fig.add_subplot(221)
    Z = least_squares_error(X, Y)
    contours = ax1.contourf(X, Y, Z, levels=20, cmap='viridis', alpha=0.7)
    setup_plot(ax1, 'Least Squares Solution')

    # Mark the least squares solution
    ax1.plot(beta_hat[0], beta_hat[1], 'r*', markersize=15)
    ax1.annotate('Least Squares\nSolution',
                 (beta_hat[0], beta_hat[1]),
                 xytext=(10, 10),
                 textcoords='offset points')
    plt.colorbar(contours, ax=ax1)

    # Ridge Regularization (L2)
    ax2 = fig.add_subplot(222)
    ax2.contourf(X, Y, Z, levels=20, cmap='viridis', alpha=0.7)

    # Draw L2 constraint (circle)
    theta = np.linspace(0, 2 * np.pi, 100)
    radius = 1
    ax2.plot(radius * np.cos(theta), radius * np.sin(theta), 'r-', linewidth=2)

    # Mark the original least squares solution
    ax2.plot(beta_hat[0], beta_hat[1], 'r*', markersize=15)

    # Ridge solution (exactly on the border)
    def ridge_objective(beta):
        # Objective: minimize distance from original solution
        # Subject to L2 norm constraint
        return np.sum((beta - beta_hat) ** 2)

    def ridge_constraint(beta):
        # L2 norm constraint
        return radius ** 2 - np.sum(beta ** 2)

    # Solve constrained optimization
    from scipy.optimize import minimize
    ridge_solution = minimize(
        ridge_objective,
        beta_hat,
        constraints={'type': 'ineq', 'fun': ridge_constraint}
    ).x

    ax2.plot(ridge_solution[0], ridge_solution[1], 'go', markersize=10)
    ax2.annotate('Ridge-Regularized\nSolution',
                 (ridge_solution[0], ridge_solution[1]),
                 xytext=(10, 10),
                 textcoords='offset points',
                 color='green')

    setup_plot(ax2, 'Ridge Regularization\n(L2 Constraint)')

    # Lasso Regularization (L1)
    ax3 = fig.add_subplot(223)
    ax3.contourf(X, Y, Z, levels=20, cmap='viridis', alpha=0.7)

    # Draw L1 constraint (diamond)
    diamond_x = [0, 1, 0, -1, 0]
    diamond_y = [1, 0, -1, 0, 1]
    ax3.plot(diamond_x, diamond_y, 'r-', linewidth=2)

    # Mark the original least squares solution
    ax3.plot(beta_hat[0], beta_hat[1], 'r*', markersize=15)

    # Lasso solution (exactly on the border)
    def lasso_objective(beta):
        # Objective: minimize distance from original solution
        # Subject to L1 norm constraint
        return np.sum((beta - beta_hat) ** 2)

    def lasso_constraint(beta):
        # L1 norm constraint
        return 1 - np.sum(np.abs(beta))

    # Solve constrained optimization
    lasso_solution = minimize(
        lasso_objective,
        beta_hat,
        constraints={'type': 'ineq', 'fun': lasso_constraint}
    ).x

    ax3.plot(lasso_solution[0], lasso_solution[1], 'go', markersize=10)
    ax3.annotate('Lasso-Regularized\nSolution',
                 (lasso_solution[0], lasso_solution[1]),
                 xytext=(10, 10),
                 textcoords='offset points',
                 color='green')

    setup_plot(ax3, 'Lasso Regularization\n(L1 Constraint)')

    # 3D Visualization of Least Squares Error Function
    ax4 = fig.add_subplot(224, projection='3d')

    # Create 3D surface
    Z = least_squares_error(X, Y)
    surf = ax4.plot_surface(X, Y, Z, cmap='viridis', alpha=0.7)

    ax4.set_title('3D Least Squares Error Surface')
    ax4.set_xlabel('β1')
    ax4.set_ylabel('β2')
    ax4.set_zlabel('Error')

    plt.tight_layout()
    plt.show()


# Call the function to create the plot
create_comprehensive_regularization_plot()