import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt


# def spline1(x, y, point):
#     f = interpolate.interp1d(x, y, kind="cubic")
#     X = np.linspace(x[0], x[-1], num=point, endpoint=True)
#     Y = f(X)
#     return X, Y


# def spline2(x, y, point):
#     f = interpolate.Akima1DInterpolator(x, y)
#     X = np.linspace(x[0], x[-1], num=point, endpoint=True)
#     Y = f(X)
#     return X, Y


def spline3(x, y, point, deg):
    tck, u = interpolate.splprep([x, y], k=deg, s=0)
    u = np.linspace(0, 1, num=point, endpoint=True)
    spline = interpolate.splev(u, tck)

    yaw = np.zeros(point)
    for i in range(1, point - 1):
        yaw[i] = np.arctan2(spline[1][i + 1] - spline[1][i - 1], spline[0][i + 1] - spline[0][i - 1]) * 180 / np.pi
    yaw[0] = yaw[1]
    yaw[-1] = yaw[-2]
    return spline[0], spline[1], yaw

if __name__ == "__main__":
    # x = [-5, 0, 1, 3, 4, 6]
    # y = [-4, 2, -2, -4, 0, 4]
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

    # a1, b1 = spline1(list(points[:, 0]), list(points[:, 1]), 100)
    # a2, b2 = spline2(list(points[:, 0]), list(points[:, 1]), 100)
    a3, b3, yaw = spline3(points[:, 0], points[:, 1], 1800, 3)
    print(yaw)
    plt.plot(points[:, 0], points[:, 1], 'ro', label="controlpoint")
    # plt.plot(a1, b1, label="interp1d")
    # plt.plot(a2, b2, label="Akima1DInterpolator")
    plt.plot(a3, b3, label="splprep")
    # print(a3)
    # print(b3)

    plt.title("spline")
    plt.xlim([-2, 8])
    plt.ylim([-2, 5])
    plt.legend(loc='lower right')
    plt.grid(which='major', color='black', linestyle='-')
    plt.grid(which='minor', color='black', linestyle='-')
    plt.xticks(list(filter(lambda x: x % 1 == 0, np.arange(-2, 8))))
    plt.yticks(list(filter(lambda x: x % 1 == 0, np.arange(-2, 5))))
    plt.savefig('hhh.png')
