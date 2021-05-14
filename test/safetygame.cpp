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

template <typename T>
using RMatrix = Matrix<T, -1, -1, RowMajor>;

double kernelF(double alpha, MatrixXd Lambda, MatrixXd zvec, MatrixXd zvecprime)
{
    return pow(alpha, 2.0) * exp(-(0.5 * (zvec - zvecprime).transpose() * (Lambda.inverse()) * (zvec - zvecprime))(0, 0));
}

MatrixXd kstarF(double alpha, MatrixXd Lambda, MatrixXd zvec, MatrixXd ZT)
{
    MatrixXd ZTprime(ZT.rows(), zvec.rows());
    ZTprime << ZT.col(0).array() - zvec(0, 0), ZT.col(1).array() - zvec(1, 0), ZT.col(2).array() - zvec(2, 0), ZT.col(3).array() - zvec(3, 0), ZT.col(4).array() - zvec(4, 0);
    return pow(alpha, 2.0) * exp((-0.5 * (((ZTprime * (Lambda.inverse())).cwiseProduct(ZTprime)).col(0) + ((ZTprime * (Lambda.inverse())).cwiseProduct(ZTprime)).col(1) + ((ZTprime * (Lambda.inverse())).cwiseProduct(ZTprime)).col(2) + ((ZTprime * (Lambda.inverse())).cwiseProduct(ZTprime)).col(3) + ((ZTprime * (Lambda.inverse())).cwiseProduct(ZTprime)).col(4))).array());
}

template <typename T>
void operation(vector<Ref<RMatrix<T>>> alpha, vector<Ref<RMatrix<T>>> Lambda, vector<Ref<RMatrix<T>>> Lambdax, vector<Ref<RMatrix<T>>> cov, Ref<RMatrix<T>> ZT, vector<Ref<RMatrix<T>>> Y, Ref<RMatrix<T>> b, vector<Ref<RMatrix<T>>> Xsafe, Ref<RMatrix<T>> Uq, double etax, vector<Ref<RMatrix<T>>> noises, double noise)
{
    cout << "Start safety game" << endl;
    vector<MatrixXd> xi(3);
    MatrixXd beta(3, 1);
    MatrixXd epsilon(3, 1);
    vector<double> c(3);
    vector<MatrixXd> ellout(3);
    MatrixXd etaxv(3, 1);
    etaxv << etax, etax, etax;

    for (int i = 0; i < 3; i++)
    {
        xi[i] = (cov[i] + noises[i](0, 0) * MatrixXd::Identity(cov[i].cols(), cov[i].rows())).inverse() * Y[i];
    }
    for (int i = 0; i < 3; i++)
    {
        beta(i, 0) = sqrt(pow(b(i, 0), 2.0) - (Y[i].transpose() * (cov[i] + noises[i](0, 0) * MatrixXd::Identity(cov[i].rows(), cov[i].cols())).inverse() * Y[i])(0, 0) + cov[i].rows());
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

    vector<vector<vector<int>>> Q(Xsafe[0].rows(), vector<vector<int>>(Xsafe[1].rows(), vector<int>(Xsafe[2].rows(), 0)));
    vector<vector<vector<int>>> Qsafe(Xsafe[0].rows(), vector<vector<int>>(Xsafe[1].rows(), vector<int>(Xsafe[2].rows(), 0)));
    for (int i = 0; i < Q.size(); i++){
        for (int j = 0; j < Q[0].size(); j++){
            for (int k = 0; k < Q[0][0].size(); k++){
                if (int((elloutmax / etax)(0, 0) + 1) <= i && i <= Q.size() - int((elloutmax / etax)(0, 0) + 1) && int((elloutmax / etax)(1, 0) + 1) <= j && j <= Q[0].size() - int((elloutmax / etax)(1, 0) + 1) && int((elloutmax / etax)(2, 0) + 1) <= k && k <= Q[0][0].size() - int((elloutmax / etax)(2, 0) + 1)){
                    Q[i][j][k] = 1;
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
    MatrixXd xrange_l(3, 1);
    MatrixXd xrange_u(3, 1);
    MatrixXd Qind_l(3, 1);
    MatrixXd Qind_u(3, 1);

    xrange_l << Xsafe[0](0, 0), Xsafe[1](0, 0), Xsafe[2](0, 0);
    xrange_u << Xsafe[0](Xsafe[0].rows() - 1, 0), Xsafe[1](Xsafe[1].rows() - 1, 0), Xsafe[2](Xsafe[2].rows() - 1, 0);

    int safeflag = 1;
    while (safeflag == 1){
        safeflag = 0;
        Qsafe = Q;
        cout << "a" << endl;
        for (int idx0 = 0; idx0 < Q.size(); idx0++)
        {
            for (int idx1 = 0; idx1 < Q[0].size(); idx1++)
            {
                for (int idx2 = 0; idx2 < Q[0][0].size(); idx2++)
                {
                    int uflag = 1;
                    for (int idu = 0; idu < Uq.rows(); idu++)
                    {
                        if (idu == Uq.rows() - 1) uflag = 0;
                        if (Q[idx0][idx1][idx2] == 1){
                            for (int i = 0; i < 3; i++)
                            {
                                xvec << Xsafe[0](idx0, 0), Xsafe[1](idx1, 0), Xsafe[2](idx2, 0);
                                zvec << Xsafe[0](idx0, 0), Xsafe[1](idx1, 0), Xsafe[2](idx2, 0), Uq(idu, 0), Uq(idu, 1);
                                kstar[i] = kstarF(alpha[i](0, 0), Lambda[i], zvec, ZT);
                                kstarstar[i] = kernelF(alpha[i](0, 0), Lambda[i], zvec, zvec);
                                means(i, 0) = (kstar[i].transpose() * xi[i])(0, 0);
                                stds(i, 0) = sqrt(kstarstar[i] - ((kstar[i].transpose() * (cov[i] + noises[i](0, 0) * MatrixXd::Identity(cov[i].cols(), cov[i].rows())).inverse()) * kstar[i])(0, 0));
                            }

                            xvecnext_l = xvec + means - (b.cwiseProduct(epsilon) + beta.cwiseProduct(stds) + noise * MatrixXd::Identity(3, 1) + etaxv);
                            xvecnext_u = xvec + means + (b.cwiseProduct(epsilon) + beta.cwiseProduct(stds) + noise * MatrixXd::Identity(3, 1) + etaxv);
                            if ((xrange_l.array() <= xvecnext_l.array()).all() == 1 && (xvecnext_u.array() <= xrange_u.array()).all() == 1){
                                Qind_l = ((xvecnext_l - xrange_l) / etax).array() + 1;
                                Qind_u = (xvecnext_u - xrange_l) / etax;

                                int qflag = 1;
                                for (int idq0 = int(Qind_l(0, 0)); idq0 <= int(Qind_u(0, 0)); idq0++){
                                    for (int idq1 = int(Qind_l(1, 0)); idq1 <= int(Qind_u(1, 0)); idq1++){
                                        for (int idq2 = int(Qind_l(2, 0)); idq2 <= int(Qind_u(2, 0)); idq2++){
                                            if (Qsafe[idq0][idq1][idq2] == 0) {
                                                qflag = 0;
                                                break;
                                            }
                                        }
                                        if (qflag == 0) break;
                                    }
                                    if (qflag == 0) break;
                                }
                                if (qflag == 1){
                                    cout << idx0 << ',' << idx1 << ',' << idx2 << endl;
                                    break;
                                }
                            }
                        }
                        if (uflag == 0){
                            Q[idx0][idx1][idx2] = 0;
                            safeflag = 1;
                            break;
                        }   
                    }
                }
            }
        }
    }

}

PYBIND11_MODULE(safetygame, m)
{
    m.doc() = "my test module";
    m.def("print_array", &operation<double>, "");
}

// MatrixXd kstarF(double alpha, MatrixXd Lambda, MatrixXd zvec, MatrixXd ZT)
// {
//     MatrixXd kstar(ZT.rows(), 1);
//     for (int i = 0; i < ZT.rows(); i++){
//         kstar(i, 0) = kernelF(alpha, Lambda, zvec, ZT.row(i).transpose());
//     }
//     return kstar;
// }


// zvec << 1, 2, 3, 4, 5;
// chrono::system_clock::time_point  starta, enda, startb, endb;

// starta = chrono::system_clock::now();
// cout << kstarF(alpha[0](0, 0), Lambda[0], zvec, ZT) << endl;
// enda = chrono::system_clock::now();
// double elapseda = chrono::duration_cast<chrono::microseconds>(enda-starta).count();
// cout << "a:" << elapseda <<"ms"<< endl;

// startb = chrono::system_clock::now();
// cout << kstarF2(alpha[0](0, 0), Lambda[0], zvec, ZT) << endl;
// endb = chrono::system_clock::now();
// double elapsedb = chrono::duration_cast<chrono::microseconds>(endb-startb).count();
// cout << "b:" << elapsedb <<"ms"<< endl;
