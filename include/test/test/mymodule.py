import numpy as np
import mymodule

x = np.arange(12).reshape(4, 3).astype(np.float64)
y = np.arange(12).reshape(3, 4).astype(np.float64)
h = np.array([[2, 1, 4], [6, 5, 2], [0, 5, 6]]).astype(np.float64)
# print("created with numpy")
# print(x)
# print()

# print("using pybind11 to print")
mymodule.print_array(x)
# print()

# x2 = mymodule.modify_array(x, 2.0)
# print("x2 newly created with pybind11")
# print(x2)
# print()

# mymodule.modify_array_inplace(x, 3.0)
# print("x modified inplace")
# print(x)

z = mymodule.modify_array_inplace2(x, y)
print(z)

zinv = mymodule.modify_array_inplace3(h)
print(zinv)
