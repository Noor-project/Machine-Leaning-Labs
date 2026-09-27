import matplotlib.pyplot as plt

help(plt.hist)
# list with 12 values
values = [1.2, 1.3, 2.2, 3.3, 2.4, 6.5, 6.6, 7.7, 8.8, 9.9, 4.2, 5.3]
plt.hist(values, bins=3)
plt.show()