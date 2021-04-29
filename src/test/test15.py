import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt


def spline3(points, point_num, deg):
    tck, u = interpolate.splprep([points[:, 0], points[:, 1]], k=deg, s=0)
    # print(u)
    u = np.linspace(0, 1, num=point_num)
    # print(tck)
    spline = interpolate.splev(u, tck)
    yaw = np.zeros(point_num)
    for i in range(1, point_num - 1):
        yaw[i] = np.arctan2(spline[1][i + 1] - spline[1][i - 1],
                            spline[0][i + 1] - spline[0][i - 1])
    yaw[0] = yaw[1]
    yaw[-1] = yaw[-2]
    # print(yaw)
    return spline[0], spline[1], yaw

if __name__ == "__main__":
    points = np.array([[0, 0],
                       [1, -0.5],
                       [2, 0],
                       [3, 0.5],
                       [4, 1.5],
                       [4.8, 1.5],
                       [5, 0.8],
                       [6, 0.5],
                       [6.5, 0],
                       [7.5, 0.5],
                       [7, 2],
                       [6, 3],
                       [5, 4],
                       [4., 2.5],
                       [3, 3],
                       [2., 3.5],
                       [1.3, 2.2],
                       [0.5, 2.],
                       [-0.1, 1],
                       [0, 0]])
    point_num = 100
    a3, b3, yaw = spline3(points, point_num, 3)
    plt.plot(points[:, 0], points[:, 1], 'ro', label="controlpoint")
    plt.plot(a3, b3, label="splprep")

    plt.title("spline")
    plt.xlim([-2, 8])
    plt.ylim([-2, 5])
    plt.legend(loc='lower right')
    plt.grid(which='major', color='black', linestyle='-')
    plt.grid(which='minor', color='black', linestyle='-')
    plt.xticks(list(filter(lambda x: x % 1 == 0, np.arange(-2, 8))))
    plt.yticks(list(filter(lambda x: x % 1 == 0, np.arange(-2, 5))))
    plt.savefig('iii.png')
