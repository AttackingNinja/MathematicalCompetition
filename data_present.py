import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator

x = np.arange(0.1, 1, 0.1)
positive_rates = np.load('positive_rates.npy', allow_pickle=True)
y = positive_rates
plt.plot(x, y)
x_major_locator = MultipleLocator(0.1)
ax = plt.gca()
ax.xaxis.set_major_locator(x_major_locator)
plt.xlim(0, 1)
plt.ylim(0, 1)
for i in range(9):
    plt.text(x[i] - 0.02, y[i] + 0.02, '%.2f' % y[i])
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.xlabel('训练集占总数据比')
plt.ylabel('测试集预测准确率')
plt.show()
plt.savefig('准确率变化图')
