#include <bits/stdc++.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <Eigen/QtAlignedMalloc>
using namespace std;
using namespace pybind11;
using namespace Eigen;

template <typename T>
using RMatrix = Matrix<T, -1, -1, RowMajor>;

template <typename T>
void print_array(RMatrix<T> a)
{
    cout << a(seq(0, 2), seq(1, 2)) << endl;
}

template <typename T>
RMatrix<T> modify_array(RMatrix<T> a, T b)
{
    return a * b;
}

template <typename T>
void modify_array_inplace(Ref<RMatrix<T>> a, T b)
{
    a = a * b;
}

template <typename T>
RMatrix<T> modify_array_inplace2(Ref<RMatrix<T>> a, Ref<RMatrix<T>> b)
{
    return a * b;
}

template <typename T>
RMatrix<T> modify_array_inplace3(Ref<RMatrix<T>> a)
{
    return a.inverse();
}

// struct buffer_info
// {
//     void *ptr;
//     ssize_t itemsize;
//     string format;
//     ssize_t ndim;
//     vector<ssize_t> shape;
//     vector<ssize_t> strides;
// };

// auto npadd(array_t<double> x, double a)
// {
//     const auto &buff_info = x.request();
//     const auto &xshape = buff_info.shape;
//     array_t<double> y{xshape};
//     for (int i = 0; i < xshape[0]; i++)
//     {
//         for (int j = 0; j < xshape[1]; j++)
//         {
//             *y.mutable_data(i, j) = *x.data(i, j) + a;
//         }
//     }
//     double p = 5.0;
//     double q = exp(p);
//     cout << q << endl;
//     cout << x.data() <<endl;
//     return y;
// }

PYBIND11_MODULE(mymodule, m)
{
    m.doc() = "my test module";
    m.def("print_array", &print_array<double>, "");
    m.def("modify_array", &modify_array<double>, "");
    m.def("modify_array_inplace", &modify_array_inplace<double>, "");
    m.def("modify_array_inplace2", &modify_array_inplace2<double>, "");
    m.def("modify_array_inplace3", &modify_array_inplace3<double>, "");
}
