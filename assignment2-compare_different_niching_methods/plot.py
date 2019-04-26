import numpy as np
import matplotlib.pyplot as plt


# plt.figure()
# x = np.linspace(3, 20, 10000)
# y = -1 * ((1/2) * x**3 - 10* x**2 + 2 * x - 1)
# p_x = np.array([19, 15, 8])
# p_y = -1 * ((1/2) * p_x**3 - 10* p_x**2 + 2 * p_x - 1)
# plt.plot(x, y)
# plt.scatter(p_x, p_y)
# k = 0
# for i, j in zip(p_x, p_y):
#     plt.annotate(s="p"+str(k), xy=(i, j))
#     k += 1
# plt.show()
#
#
# plt.figure()
# x = np.linspace(-6, 20, 10000)
# y = -1 * ((1/2) * x**3 - 10* x**2 + 2 * x - 1)
# p_x = np.array([19, 15, 8])
# p_y = -1 * ((1/2) * p_x**3 - 10* p_x**2 + 2 * p_x - 1)
# plt.plot(x, y)
# plt.scatter(p_x, p_y)
# k = 0
# for i, j in zip(p_x, p_y):
#     plt.annotate(s="p"+str(k), xy=(i, j))
#     k += 1
# plt.vlines(p_x[0], -50, p_y[0], colors = "c", linestyles = "dashed")
# plt.vlines(p_x[1], -50, p_y[1], colors = "c", linestyles = "dashed")
# plt.vlines(p_x[2], -50, p_y[2], colors = "c", linestyles = "dashed")
# plt.vlines(0, -50, 0, colors = "c", linestyles = "dashed")
# plt.annotate(s="c1", xy=(-5, 50))
# plt.annotate(s="c2", xy=(12, 50))
# plt.annotate(s="c3", xy=(17, 50))
# plt.show()

plt.figure()
p_x = np.array([1, 2, 1, 2, 0, 0, 3, 3])
p_y = np.array([1, 2, 2, 1, 0, 3, 0, 3])
x1 = np.array([1, 1, 2, 2, 1])
y1 = np.array([1, 2, 2, 1, 1])
x2 = np.array([0, 0, 3, 3, 0])
y2 = np.array([0, 3, 3, 0, 0])
plt.plot(x1, y1)
plt.plot(x2, y2)
plt.scatter(p_x, p_y)
plt.annotate(s="p1", xy=(p_x[0], p_y[0]))
plt.annotate(s="p2", xy=(p_x[1], p_y[1]))
plt.annotate(s="p3", xy=(p_x[2], p_y[2]))
plt.annotate(s="p4", xy=(p_x[3], p_y[3]))
plt.annotate(s="p5", xy=(p_x[4], p_y[4]))
plt.annotate(s="p6", xy=(p_x[5], p_y[5]))
plt.annotate(s="p7", xy=(p_x[6], p_y[6]))
plt.annotate(s="p8", xy=(p_x[7], p_y[7]))

plt.arrow(1, 1, 1, 1, length_includes_head=True,
             head_width=0.1, head_length=0.1, fc='r', ec='b')
plt.arrow(2, 2, 1, 1, length_includes_head=True,
             head_width=0.1, head_length=0.1, fc='r', ec='b')
plt.show()

