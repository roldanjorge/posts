import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

# Create a 3D figure
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Generate data for the plane
x = np.linspace(-5, 5, 20)
y = np.linspace(-5, 5, 20)
x, y = np.meshgrid(x, y)
z = 2 * x + 3 * y + 1   # Example plane equation: z = 2x + 3y + 1

# Plot the plane
ax.plot_surface(x, y, z, alpha=0.5, cmap='viridis', edgecolor='k')

# Generate some sample data points
np.random.seed(42)
x_data = np.random.uniform(-5, 5, 10)
y_data = np.random.uniform(-5, 5, 10)
z_data = 2 * x_data + 3 * y_data + 1 + np.random.normal(scale=2, size=10)  # Add noise

# Plot the data points
ax.scatter(x_data, y_data, z_data, color='red', s=50, label='Data points')

# Draw residual lines (vertical lines from the data point to the plane)
for i in range(len(x_data)):
    z_plane = 2 * x_data[i] + 3 * y_data[i] + 1  # z value on the plane at (x_data, y_data)
    ax.plot(
        [x_data[i], x_data[i]],  # x-coordinates
        [y_data[i], y_data[i]],  # y-coordinates
        [z_data[i], z_plane],    # z-coordinates
        color='gray', linestyle='--', linewidth=1
    )

# Add labels
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D RSS Visualization')

# Add legend
ax.legend()

# Show the plot
plt.show()
