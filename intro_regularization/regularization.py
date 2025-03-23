import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

# Create a 3D figure
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Generate data for the plane
x1 = np.linspace(-5, 5, 20)
x2 = np.linspace(-5, 5, 20)
x1, x2 = np.meshgrid(x1, x2)
y = 2 * x1 + 3 * x2 + 1   # Example plane equation: y = 2x1 + 3x2 + 1

# Plot the plane
ax.plot_surface(x1, x2, y, alpha=0.5, cmap='viridis', edgecolor='k')

# Generate some sample data points
np.random.seed(42)
x1_data = np.random.uniform(-5, 5, 10)
x2_data = np.random.uniform(-5, 5, 10)
y_data = 3 * x1_data + 5 * x2_data + 1 + np.random.normal(scale=4, size=10)  # Add noise

# Plot the data points
ax.scatter(x1_data, x2_data, y_data, color='red', s=50, label='Data points')

# Draw residual lines (vertical lines from the data point to the plane)
for i in range(len(x1_data)):
    y_plane = 2 * x1_data[i] + 3 * x2_data[i] + 1  # z value on the plane at (x_data, y_data)
    ax.plot(
        [x1_data[i], x1_data[i]],  # x-coordinates
        [x2_data[i], x2_data[i]],  # y-coordinates
        [y_data[i], y_plane],    # z-coordinates
        color='black', linestyle='-', linewidth=2
    )

# Add labels
ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_zlabel('Y')
ax.set_title('3D RSS Visualization')

# Add legend
ax.legend()

# Save the plot as an image (change format if needed)
plt.savefig('rss_visualization.png', dpi=300, bbox_inches='tight')  # Save as PNG with high resolution

# Show the plot
plt.show()
