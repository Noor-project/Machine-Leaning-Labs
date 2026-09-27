import matplotlib.pyplot as plt
import numpy as np

# Sample data with an outlier
data = [10, 12, 11, 13, 12, 14, 13, 100]  # 100 is an outlier

# Plot boxplot
plt.boxplot(data)
plt.title("Outlier Visualization")
plt.show()