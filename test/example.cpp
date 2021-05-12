#include <bits/stdc++.h>
// #include <pybind11/pybind11.h>
// #include <pybind11/numpy.h>
// #include <pybind11/eigen.h>
// #include <Eigen/QtAlignedMalloc>
using namespace std;
// using namespace pybind11;
// using namespace Eigen;
using ll = long long;

void init()
{
    ios::sync_with_stdio(false);
    cin.tie(0);
    cout << fixed << setprecision(20);
}

int main()
{
    init();
    chrono::system_clock::time_point start, end;

    int a = 0;
    start = chrono::system_clock::now();
    for (ll i = 0; i < 10000000000; i++){
        a++;
    }
    end = std::chrono::system_clock::now();
    double elapsed = chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    cout << elapsed << endl;
}
