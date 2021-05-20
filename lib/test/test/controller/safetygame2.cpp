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

double kernelMetric(double alpha, MatrixXd Lambda, MatrixXd trlen){
    return sqrt(2 * pow(alpha, 2.0) - 2 * kernelF(alpha, Lambda, trlen, MatrixXd::Zero(3, 1)));
}

int contractiveF(double alpha, MatrixXd Lambda, MatrixXd trlen, double epsilon, double gamma){
    double kmd = kernelMetric(alpha, Lambda, trlen);
    // cout << "a" << endl;
    // cout << kmd << endl;
    // cout << epsilon - gamma << endl;
    // cout << epsilon << endl;
    // cout << gamma << endl;
    if (kmd <= epsilon - gamma){
        return 1;
    }else{
        return 0;
    }
}

int safeF(vector<vector<vector<int>>>Qsafe, MatrixXd Qind_l, MatrixXd Qind_u){
    for (int idq0 = int(Qind_l(0, 0)); idq0 <= int(Qind_u(0, 0)); idq0++){
        for (int idq1 = int(Qind_l(1, 0)); idq1 <= int(Qind_u(1, 0)); idq1++){
            for (int idq2 = int(Qind_l(2, 0)); idq2 <= int(Qind_u(2, 0)); idq2++){
                if (Qsafe[idq0][idq1][idq2] == 0) return 0;
            }
        }
    }
    return 1;
}

template <typename T>
vector<vector<vector<int>>> operation(vector<vector<vector<int>>> Q, Ref<RMatrix<T>> Qind, double alpha, Ref<RMatrix<T>> Lambda, Ref<RMatrix<T>> Lambdax, Ref<RMatrix<T>> cov, double noises, Ref<RMatrix<T>> ZT, Ref<RMatrix<T>> Y, Ref<RMatrix<T>> b, vector<Ref<RMatrix<T>>> Xqlist, Ref<RMatrix<T>> Uq, double etax, double epsilon, double gamma, Ref<RMatrix<T>> y_mean, Ref<RMatrix<T>> y_std)
{
    cout << "start safety game." << endl;
    MatrixXd xvec(3, 1);
    MatrixXd zvec(5, 1);
    double kstarstar;
    MatrixXd kstar(Y.rows(), 1);
    MatrixXd means(3, 1);
    MatrixXd stds(3, 1);
    MatrixXd xvecnext_l(3, 1);
    MatrixXd xvecnext_u(3, 1);
    MatrixXd Qind_l(3, 1);
    MatrixXd Qind_u(3, 1);
    MatrixXd xi(Y.rows(), 3);
    MatrixXd beta(3, 1);
    MatrixXd etaxv(3, 1);
    MatrixXd xrange_l(3, 1);
    MatrixXd xrange_u(3, 1);
    MatrixXd trlen(3, 1);
    vector<vector<vector<int>>> Qsafe;
    Qsafe = Q;
    etaxv << etax, etax, etax;
    xrange_l << Xqlist[0](0, 0), Xqlist[1](0, 0), Xqlist[2](0, 0);
    xrange_u << Xqlist[0](Xqlist[0].rows() - 1, 0), Xqlist[1](Xqlist[1].rows() - 1, 0), Xqlist[2](Xqlist[2].rows() - 1, 0);
    for (int i = 0; i < 3; i++)
    {
        xi.col(i) = cov.inverse() * Y.col(i);
        beta(i, 0) = sqrt(pow(b(i, 0), 2.0) - (Y.col(i).transpose() * cov.inverse() * Y.col(i))(0, 0) + cov.rows());
    }

    cout << beta << endl;

    for (int idq = 0; idq < Qind.rows(); idq++){
        int uflag = 1;
        for (int idu = 0; idu < Uq.rows(); idu++){
            if (idu == Uq.rows() - 1) uflag = 0;
            xvec << Xqlist[0](Qind(idq, 0), 0), Xqlist[1](Qind(idq, 1), 0), Xqlist[2](Qind(idq, 2), 0);
            zvec << Xqlist[0](Qind(idq, 0), 0), Xqlist[1](Qind(idq, 1), 0), Xqlist[2](Qind(idq, 2), 0), Uq(idu, 0), Uq(idu, 1);
            kstar = kstarF(alpha, Lambda, zvec, ZT);
            // kstarstar = kernelF(alpha, Lambda, zvec, zvec);
            
            means = y_std.cwiseProduct((kstar.transpose() * xi).transpose()) + y_mean;
            stds = y_std * sqrt(pow(alpha, 2.0) + noises - (kstar.transpose() * cov.inverse() * kstar)(0, 0));
            trlen = epsilon * b.cwiseProduct(y_std) + beta.cwiseProduct(stds).cwiseProduct(y_std) + etaxv;

            xvecnext_l = xvec + means - trlen - 0.1 * MatrixXd::Ones(3, 1);
            xvecnext_u = xvec + means + trlen + 0.1 * MatrixXd::Ones(3, 1);

            int qflag = 0;
            if ((xrange_l.array() <= xvecnext_l.array()).all() == 1 && (xvecnext_u.array() <= xrange_u.array()).all() == 1){
                Qind_l = ((xvecnext_l - xrange_l) / etax).array() + 1;
                Qind_u = (xvecnext_u - xrange_l) / etax;
                qflag = safeF(Qsafe, Qind_l, Qind_u);
            }
            if (qflag == 1){
                cout << idq << "/" << Qind.rows() << endl;
                break;
            }
            if (uflag == 0){
                Q[Qind(idq, 0)][Qind(idq, 1)][Qind(idq, 2)] = 0;
            }
        }
    }
    return Q;
}

PYBIND11_MODULE(safetygame2, m)
{
    m.doc() = "my test module";
    m.def("operation", &operation<double>, "");
}


