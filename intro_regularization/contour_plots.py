import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

# Create grid for beta values
beta1 = np.linspace(-1.5, 1.5, 100)
beta2 = np.linspace(-1.5, 1.5, 100)
B1, B2 = np.meshgrid(beta1, beta2)

# Define the least squares error contours
loss = B1**2 + 0.5 * B2**2

# Plot least squares error contours
plt.figure(figsize=(12, 6))

# Ridge Regularization Plot
plt.subplot(1, 2, 1)
plt.contour(B1, B2, loss, levels=np.logspace(-1, 1, 10), cmap="Blues")
circle = plt.Circle((0, 0), radius=1, color='red', fill=False, label=r'$||\beta||_2 \leq t$')
plt.gca().add_patch(circle)
plt.axhline(0, color='gray', linestyle='--', linewidth=1)
plt.axvline(0, color='gray', linestyle='--', linewidth=1)
plt.title('Ridge Regularization')
plt.xlabel(r'$\beta_1$')
plt.ylabel(r'$\beta_2$')
plt.legend()

# Lasso Regularization Plot
plt.subplot(1, 2, 2)
plt.contour(B1, B2, loss, levels=np.logspace(-1, 1, 10), cmap="Blues")
diamond = plt.plot([1, 0, -1, 0, 1], [0, 1, 0, -1, 0], color='red', label=r'$||\beta||_1 \leq t$')
plt.axhline(0, color='gray', linestyle='--', linewidth=1)
plt.axvline(0, color='gray', linestyle='--', linewidth=1)
plt.title('Lasso Regularization')
plt.xlabel(r'$\beta_1$')
plt.ylabel(r'$\beta_2$')
plt.legend()

plt.tight_layout()
plt.show()
