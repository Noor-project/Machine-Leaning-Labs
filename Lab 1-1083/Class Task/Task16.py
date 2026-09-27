import matplotlib.pyplot as plt

year = [1994, 1995, 1998, 2000]  # data points
population = [2.59, 3.69, 5.33, 6.77]  # data points
plt.plot(year, population)  # plots the data on specified co-ordinates in background
plt.show()  # visualises the graph
plt.scatter(year, population)
plt.show()