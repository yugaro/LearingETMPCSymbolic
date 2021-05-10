#include <bits/stdc++.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <Eigen/Dense>
using namespace std;
using namespace pybind11;
using namespace Eigen;

template <typename T>
using RMatrix = Matrix<T, -1, -1, RowMajor>;

template <typename T>
void print_array(Ref<RMatrix<T>> a, Ref<RMatrix<T>> Lambdax0, Ref<RMatrix<T>> Lambdax1, Ref<RMatrix<T>> Lambdax2, Ref<RMatrix<T>> cov0, Ref<RMatrix<T>> cov1, Ref<RMatrix<T>> cov2, Ref<RMatrix<T>> Y0, Ref<RMatrix<T>> Y1, Ref<RMatrix<T>> Y2)
{
    // cout << a << endl;
    // cout << Lambdax0 << endl;
    // cout << Lambdax1 << endl;
    // cout << Lambdax2 << endl;
    // cout << cov0 << endl;
    // cout << cov1 << endl;
    // cout << cov2 << endl;
    cout << cov0.inverse() << endl;
}

PYBIND11_MODULE(safetygame, m)
{
    m.doc() = "my test module";
    m.def("print_array", &print_array<double>, "");
}
