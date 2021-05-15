import numpy as np
import cppmod

x = np.arange(2 * 3).reshape((2, 3))
x2 = cppmod.npadd(x, 10)
# print(x2)
