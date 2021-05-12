#include <bits/stdc++.h>
#include <valarray>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>
#include <Eigen/Dense>
#include <boost/multi_array.hpp>

using namespace std;
using namespace pybind11;
using namespace Eigen;
using namespace boost;

using ma_type = multi_array<int, 3>;
using ma_index = ma_type::index;
using range = multi_array_types::index_range;

template <typename T>
using RMatrix = Matrix<T, -1, -1, RowMajor>;

double kernelF(double alpha, MatrixXd Lambda, MatrixXd zvec, MatrixXd zvecprime)
{
    return pow(alpha, 2.0) * exp(-(0.5 * (zvec - zvecprime).transpose() * (Lambda.inverse()) * (zvec - zvecprime))(0, 0));
}

MatrixXd kstarF(double alpha, MatrixXd Lambda, MatrixXd zvec, MatrixXd ZT)
{
    MatrixXd kstar(ZT.rows(), 1);
    for (int i = 0; i < ZT.rows(); i++){
        kstar(i, 0) = kernelF(alpha, Lambda, zvec, ZT.row(i).transpose());
    }
    return kstar;
}

template <typename T>
void operation(vector<Ref<RMatrix<T>>> alpha, vector<Ref<RMatrix<T>>> Lambda, vector<Ref<RMatrix<T>>> Lambdax, vector<Ref<RMatrix<T>>> cov, Ref<RMatrix<T>> ZT, vector<Ref<RMatrix<T>>> Y, Ref<RMatrix<T>> b, double noise, vector<Ref<RMatrix<T>>> Xsafe, Ref<RMatrix<T>> Uq, double etax, vector<Ref<RMatrix<T>>> noises)
{
    vector<MatrixXd> xi(3);
    MatrixXd beta(3, 1);
    MatrixXd epsilon(3, 1);
    vector<double> c(3);
    vector<MatrixXd> ellout(3);
    MatrixXd etaxv(3, 1);
    etaxv << etax, etax, etax;
    for (int i = 0; i < 3; i++)
    {
        xi[i] = (cov[i] + pow(noises[i](0, 0), 2.0) * MatrixXd::Identity(cov[i].rows(), cov[i].cols())).inverse() * Y[i];
    }
    for (int i = 0; i < 3; i++)
    {
        beta(i, 0) = sqrt(pow(b(i, 0), 2.0) - (Y[i].transpose() * (cov[i] + pow(noises[i](0, 0), 2.0) * MatrixXd::Identity(cov[i].rows(), cov[i].cols())).inverse() * Y[i])(0, 0) + cov[i].rows());
    }
    for (int i = 0; i < 3; i++){
        epsilon(i, 0) = sqrt(2 * pow(alpha[i](0, 0), 2.0) - 2 * kernelF(alpha[i](0, 0), Lambdax[i], etaxv, MatrixXd::Zero(3, 1)));
    }
    for (int i = 0; i < 3; i++){
        c[i] = sqrt(2 * log(2 * pow(alpha[i](0, 0), 2.0) / (2 * pow(alpha[i](0, 0), 2.0) - pow(epsilon(i, 0), 2.0))));
    }
    for (int i = 0; i < 3; i++){
        ellout[i] = c[i] * (Lambdax[i].diagonal().array().sqrt());
    }

    double ellout0max = ellout[0](0, 0), ellout1max = ellout[0](1, 0), ellout2max = ellout[0](2, 0);
    for (int i = 1; i < 3; i++){
        if (ellout0max < ellout[i](0, 0)) ellout0max = ellout[i](0, 0);
        if (ellout1max < ellout[i](1, 0)) ellout1max = ellout[i](1, 0);
        if (ellout2max < ellout[i](2, 0)) ellout2max = ellout[i](2, 0);
    }
    MatrixXd elloutmax(3, 1);
    elloutmax << ellout0max, ellout1max, ellout2max;

    ma_type Q(extents[Xsafe[0].rows()][Xsafe[1].rows()][Xsafe[2].rows()]);

    for (int i = 0; i < Q.shape()[0]; i++){
        for (int j = 0; j < Q.shape()[1]; j++){
            for (int k = 0; k < Q.shape()[2]; k++){
                if (int((elloutmax / etax)(0, 0) + 1) <= i && i <= Q.shape()[0] - int((elloutmax / etax)(0, 0) + 1) && int((elloutmax / etax)(1, 0) + 1) <= j && j <= Q.shape()[1] - int((elloutmax / etax)(1, 0) + 1) && int((elloutmax / etax)(2, 0) + 1) <= k && k <= Q.shape()[2] - int((elloutmax / etax)(2, 0) + 1)){
                    Q[i][j][k] = 1;
                }else{
                    Q[i][j][k] = 0;
                }
            }
        }
    }

    MatrixXd xvec(3, 1);
    MatrixXd zvec(5, 1);
    vector<double> kstarstar(3);
    vector<MatrixXd> kstar(3);
    MatrixXd means(3, 1);
    MatrixXd stds(3, 1);
    MatrixXd xvecnext_l(3, 1);
    MatrixXd xvecnext_u(3, 1);


    for (int idx0 = 0; idx0 < Q.shape()[0]; idx0++)
    {
        for (int idx1 = 0; idx1 < Q.shape()[1]; idx1++)
        {
            for (int idx2 = 0; idx2 < Q.shape()[2]; idx2++)
            {
                for (int idu = 0; idu < Uq.rows(); idu++)
                {
                    if(Q[idx0][idx1][idx2] == 1){
                        for (int i = 0; i < 3; i++)
                        {
                            xvec << Xsafe[0](idx0, 0), Xsafe[1](idx1, 0), Xsafe[2](idx2, 0);
                            zvec << Xsafe[0](idx0, 0), Xsafe[1](idx1, 0), Xsafe[2](idx2, 0), Uq(idu, 0), Uq(idu, 1);
                            kstar[i] = kstarF(alpha[i](0, 0), Lambda[i], zvec, ZT);
                            kstarstar[i] = kernelF(alpha[i](0, 0), Lambda[i], zvec, zvec);
                            means(i, 0) = (kstar[i].transpose() * xi[i])(0, 0);
                            stds(i, 0) = sqrt(kstarstar[i] - ((kstar[i].transpose() * (cov[i] + pow(noises[i](0, 0), 2.0) * MatrixXd::Identity(cov[i].cols(), cov[i].rows())).inverse()) * kstar[i])(0, 0));
                        }

                        xvecnext_l = xvec + means - (b.cwiseProduct(epsilon) + beta.cwiseProduct(stds) + etaxv);
                        xvecnext_u = xvec + means + (b.cwiseProduct(epsilon) + beta.cwiseProduct(stds) + etaxv);

                        cout << zvec << endl;
                        cout << means << endl;
                        cout << stds << endl;

                        if (idu ==1) return;

                    }
                }
            }
        }
    }

    // cout << 'a' << endl;
    // for (ma_index i = 0; i < Xsafe[0].cols(); i++)
    // {
    //     for (ma_index j = 0; j < Xsafe[1].cols(); j++)
    //     {
    //         for (ma_index k = 0; k < Xsafe[2].cols(); k++)
    //         {
    //             my_array[i][j][k] = i + j + k;
    //         }
    //     }
    // }
    // cout << my_array[0][3][8] << endl;
    // ma_type::array_view<3>::type my_view =
    //     my_array[indices[range(0, 2)][range(1, 3)][range(2, 8)]];
    // cout << my_view.shape()[0] << endl;
    // cout << my_view.shape()[1] << endl;
    // cout << my_view.shape()[2] << endl;
    // cout << minmax(my_view.begin(), my_view.end()) << endl;

   
}

PYBIND11_MODULE(safetygame, m)
{
    m.doc() = "my test module";
    m.def("print_array", &operation<double>, "");
}
