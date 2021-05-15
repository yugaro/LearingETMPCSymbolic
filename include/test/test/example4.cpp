#include <pybind11/pybind11.h>
#include <bits/stdc++.h>
using namespace std;
using namespace pybind11;

int add(int i, int j)
{
    return i + j;
}

PYBIND11_MODULE(example4, m)
{
    m.doc() = "pybind11 example plugin"; // optional module docstring

    m.def("add", &add, "A function which adds two numbers");
}
