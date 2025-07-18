# Re-import necessary modules after state reset
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Define the range of personal and social coefficients
personal_coeffs = [2.5, 2.0, 1.5, 1.0, 0.5]
social_coeffs = [0.5, 1.0, 1.5, 2.0, 2.5]

# Mock iteration count results matrix (example values for demonstration)
iteration_matrix = np.array([
    [134, 121, 108, 95, 87],
    [141, 125, 113, 98, 92],
    [155, 137, 121, 106, 99],
    [167, 149, 130, 115, 102],
    [178, 161, 140, 120, 110]
])

# Plotting the heatmap (with iteration counts)
plt.figure(figsize=(8, 6))
ax = sns.heatmap(iteration_matrix, annot=True, fmt="d", cmap="YlGnBu", xticklabels=social_coeffs, yticklabels=personal_coeffs)
plt.title("Iterations to Convergence\n(Personal Coefficient vs Social Coefficient)")
plt.xlabel("Social Coefficient")
plt.ylabel("Personal Coefficient")
plt.tight_layout()
plt.show()
