import sys
import matplotlib.pyplot as plt

data = []

for line in sys.stdin:
    new_list = [float(val) for val in line.split()]
    data.append(new_list)

real = data[0]
out1 = data[1]
out2 = data[2]
hist1 = data[3]
hist2 = data[4]
x = list(range(0, len(real)))

plt.plot(x, real, "r--")
plt.plot(x, out1, "g--")
plt.plot(x, out2, "b--") 
plt.show()


plt.plot(list(range(0, len(hist1))), [0]*len(hist1), 'r--')
plt.plot(list(range(0, len(hist1))), hist1, "g--")
plt.plot(list(range(0, len(hist1))), hist2, "b--")
plt.xscale('log')
plt.show()
