import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from scipy.interpolate import make_interp_spline  # For curve smoothing

results_dir_rrt = "rrt_simulation_results_csv"  # Update with your results directory

all_results_rrt = []
for filename in os.listdir(results_dir_rrt):
    if filename.endswith(".csv"):
        df = pd.read_csv(os.path.join(results_dir_rrt, filename))
        all_results_rrt.append(df)

results_df_rrt = pd.concat(all_results_rrt)

# --- Outlier Removal ---
def remove_outliers_iqr(df, column_name):
    """Removes outliers from a DataFrame column using the IQR method.

    Args:
        df: The pandas DataFrame.
        column_name: The name of the column to remove outliers from.

    Returns:
        A new DataFrame with the outliers removed.
    """
    q1 = df[column_name].quantile(0.25)
    q3 = df[column_name].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    return df[(df[column_name] >= lower_bound) & (df[column_name] <= upper_bound)]

# Remove outliers for each group (number of drones) based on 'iterations'
filtered_results_df = pd.DataFrame()
for num_drones, group_data in results_df_rrt.groupby("num_drones"):
    filtered_group = remove_outliers_iqr(group_data, "iterations")
    filtered_results_df = pd.concat([filtered_results_df, filtered_group])

# Calculate average iterations and success rate for each number of drones
avg_iterations = filtered_results_df.groupby("num_drones")["iterations"].mean()
success_rate = filtered_results_df.groupby("num_drones")["target_found"].mean()

# --- Smoothing the Average Iterations Curve ---
# Create a new x-axis with more points for smoother curve
xnew = np.linspace(avg_iterations.index.min(), avg_iterations.index.max(), 300)

# Interpolate the data using a spline
spl = make_interp_spline(avg_iterations.index, avg_iterations.values, k=3)  # Cubic spline
smooth_avg_iterations = spl(xnew)

# Plot the results
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.scatter(filtered_results_df["num_drones"], filtered_results_df["iterations"], alpha=0.5, label="Individual Runs")
plt.plot(avg_iterations.index, avg_iterations.values, marker='o', color='red', label="Average Iterations")
#plt.plot(xnew, smooth_avg_iterations, color='red', label="Smoothed Avg. Iterations")
plt.xlabel("Number of Drones")
plt.ylabel("Average Iterations to Find Target")
plt.title("RRT Performance vs. Number of Drones")

plt.subplot(1, 2, 2)
plt.plot(success_rate.index, success_rate.values, marker='o', color='green')
plt.xlabel("Number of Drones")
plt.ylabel("Success Rate")
plt.title("RRT Success Rate vs. Number of Drones")

plt.tight_layout()
plt.show()
