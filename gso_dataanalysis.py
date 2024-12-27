import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from scipy.interpolate import make_interp_spline  # For curve smoothing

results_dir_gso = "gso_simulation_results_csv"  # Update with your results directory

all_results_gso = []
for filename in os.listdir(results_dir_gso):
    if filename.endswith(".csv"):
        df = pd.read_csv(os.path.join(results_dir_gso, filename))
        all_results_gso.append(df)

results_df_gso = pd.concat(all_results_gso)

# Calculate average iterations and success rate for each number of drones
avg_iterations = results_df_gso.groupby("num_drones")["iterations"].mean()
success_rate = results_df_gso.groupby("num_drones")["target_found"].mean()

# --- Smoothing the Average Iterations Curve ---
# Create a new x-axis with more points for smoother curve
xnew = np.linspace(avg_iterations.index.min(), avg_iterations.index.max(), 300)

# Interpolate the data using a spline
spl = make_interp_spline(avg_iterations.index, avg_iterations.values, k=3)  # Cubic spline
smooth_avg_iterations = spl(xnew)

# Plot the results
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.scatter(results_df_gso["num_drones"], results_df_gso["iterations"], alpha=0.5, label="Individual Runs")
plt.plot(avg_iterations.index, avg_iterations.values, marker='o', color='red', label="Average Iterations")
#plt.plot(xnew, smooth_avg_iterations, color='red', label="Smoothed Avg. Iterations")
plt.xlabel("Number of Drones")
plt.ylabel("Average Iterations to Find Target")
plt.title("GSO Performance vs. Number of Drones")

plt.subplot(1, 2, 2)
plt.plot(success_rate.index, success_rate.values, marker='o', color='green')
plt.xlabel("Number of Drones")
plt.ylabel("Success Rate")
plt.title("GSO Success Rate vs. Number of Drones")

plt.tight_layout()
plt.show()