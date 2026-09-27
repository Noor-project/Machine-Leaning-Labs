import matplotlib.pyplot as plt

year = [1994, 1995, 1998, 2000]  # data points
population = [2.59, 3.69, 5.33, 6.77]  # data points
plt.plot(year, population, 'r', label='Population 1', linewidth=5)  # plots the data on specified co-ordinates in background
pop2 = [4.44, 3.22, 5.55, 6.88]
plt.plot(year, pop2, 'c', label='Population 2', linewidth=5)  # plots the data and assign label to it
plt.xlabel('Independent var')  # specifies xlabel
plt.ylabel('Dependent var')  # specifies ylabel
plt.title('Interesting Graph')  # specifies title
plt.legend()  # visualises labels specified to the data
plt.grid(True, color='k')
plt.fill_between(year, population, 0, color='green')  # fills the specified color below the data points
plt.show()