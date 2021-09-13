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

double kernelMetric(double alpha, MatrixXd Lambda, MatrixXd xqout, MatrixXd xqin)
{
    return sqrt(2 * pow(alpha, 2.0) - 2 * kernelF(alpha, Lambda, xqout, xqin));
}

int contractiveF(double alpha, MatrixXd Lambda, MatrixXd xqout, MatrixXd xqin, double gamma)
{
    if (kernelMetric(alpha, Lambda, xqout, xqin) > gamma) return 1;
    return 0;
}

int safeF(vector<vector<vector<int>>> Qsafe, MatrixXd Qind_lout, MatrixXd Qind_uout, MatrixXd Qind_lin, MatrixXd Qind_uin, MatrixXd Xqlist0, MatrixXd Xqlist1, MatrixXd Xqlist2, double alpha, MatrixXd Lambda, double gamma)
{
    MatrixXd xqout(3, 1);
    MatrixXd xqin(3, 1);
    double conx0;
    double conx1;
    double conx2;

    for (int idq0 = int(Qind_lout(0, 0)); idq0 <= int(Qind_uout(0, 0)); idq0++)
    {
        int recflag0 = 0;
        if (idq0 < int(Qind_lin(0, 0)))
        {
            conx0 = Xqlist0(int(Qind_lin(0, 0)), 0);
            recflag0 = 1;
        }
        else if (idq0 > int(Qind_uin(0, 0)))
        {
            conx0 = Xqlist0(int(Qind_uin(0, 0)), 0);
            recflag0 = 1;
        }
        for (int idq1 = int(Qind_lout(1, 0)); idq1 <= int(Qind_uout(1, 0)); idq1++)
        {
            int recflag1 = 0;
            if (recflag0 == 1)
            {
                if (idq1 < int(Qind_lin(1, 0)))
                {
                    conx1 = Xqlist1(int(Qind_lin(1, 0)), 0);
                    recflag1 = 1;
                }
                else if (idq1 > int(Qind_uin(1, 0)))
                {
                    conx1 = Xqlist1(int(Qind_uin(1, 0)), 0);
                    recflag1 = 1;
                }
            }
            for (int idq2 = int(Qind_lout(2, 0)); idq2 <= int(Qind_uout(2, 0)); idq2++)
            {
                int recflag2 = 0;
                if (recflag0 * recflag1 == 1)
                {
                    if (idq2 < int(Qind_lin(2, 0)))
                    {
                        conx2 = Xqlist2(int(Qind_lin(2, 0)), 0);
                        recflag2 = 1;
                    }
                    else if (idq2 > int(Qind_uin(2, 0)))
                    {
                        conx2 = Xqlist2(int(Qind_uin(2, 0)), 0);
                        recflag2 = 1;
                    }
                }
                if (Qsafe[idq0][idq1][idq2] == 0)
                {
                    if (recflag0 * recflag1 * recflag2 == 1)
                    {
                        xqout << Xqlist0(idq0, 0), Xqlist1(idq1, 0), Xqlist2(idq2, 0);
                        xqin << conx0, conx1, conx2;
                        if (contractiveF(alpha, Lambda, xqout, xqin, gamma) == 1) continue;
                    }
                    return 0;
                }
            }
        }
    }
    return 1;
}

template <typename T>
tuple<vector<vector<vector<int>>>, RMatrix<T>> operation(vector<vector<vector<int>>> Q, Ref<RMatrix<T>> Qind, double alpha, Ref<RMatrix<T>> Lambda, Ref<RMatrix<T>> Lambdax, Ref<RMatrix<T>> cov, double noises, Ref<RMatrix<T>> ZT, Ref<RMatrix<T>> Y, Ref<RMatrix<T>> b, vector<Ref<RMatrix<T>>> Xqlist, Ref<RMatrix<T>> Uq, Ref<RMatrix<T>> etax_v, double epsilon, double gamma, Ref<RMatrix<T>> ellin, int flag_refcon, Ref<RMatrix<T>> y_mean, Ref<RMatrix<T>> y_std)
{
    cout << "start safety game." << endl;
    MatrixXd xvec(3, 1);
    MatrixXd zvec(5, 1);
    MatrixXd kstar(Y.rows(), 1);
    MatrixXd means(3, 1);
    MatrixXd stds;
    MatrixXd xvecnext(3, 1);
    MatrixXd xvecnext_lout(3, 1);
    MatrixXd xvecnext_uout(3, 1);
    MatrixXd xvecnext_lin(3, 1);
    MatrixXd xvecnext_uin(3, 1);
    MatrixXd Qind_lout(3, 1);
    MatrixXd Qind_uout(3, 1);
    MatrixXd Qind_lin(3, 1);
    MatrixXd Qind_uin(3, 1);
    MatrixXd xi(Y.rows(), 3);
    MatrixXd beta(3, 1);
    MatrixXd xrange_l(3, 1);
    MatrixXd xrange_u(3, 1);
    MatrixXd trlen(3, 1);
    vector<vector<vector<int>>> Qsafe;
    MatrixXd Cs(Qind.rows(), 2);

    Qsafe = Q;
    xrange_l << Xqlist[0](0, 0), Xqlist[1](0, 0), Xqlist[2](0, 0);
    xrange_u << Xqlist[0](Xqlist[0].rows() - 1, 0), Xqlist[1](Xqlist[1].rows() - 1, 0), Xqlist[2](Xqlist[2].rows() - 1, 0);
    for (int i = 0; i < 3; i++)
    {
        xi.col(i) = cov.inverse() * Y.col(i);
        beta(i, 0) = sqrt(pow(b(i, 0), 2.0) - (Y.col(i).transpose() * cov.inverse() * Y.col(i))(0, 0) + cov.rows());
    }

    beta << 1, 1, 1;

    if (flag_refcon == 0){
        for (int idq = 0; idq < Qind.rows(); idq++)
        {
            int uflag = 1;
            for (int idu = 0; idu < Uq.rows(); idu++)
            {
                if (idu == Uq.rows() - 1)
                    uflag = 0;
                xvec << Xqlist[0](Qind(idq, 0), 0), Xqlist[1](Qind(idq, 1), 0), Xqlist[2](Qind(idq, 2), 0);
                zvec << Xqlist[0](Qind(idq, 0), 0), Xqlist[1](Qind(idq, 1), 0), Xqlist[2](Qind(idq, 2), 0), Uq(idu, 0), Uq(idu, 1);

                kstar = kstarF(alpha, Lambda, zvec, ZT);
                means = ((kstar.transpose() * xi).transpose()).cwiseProduct(y_std) + y_mean;
                stds = sqrt(pow(alpha, 2.0) - (kstar.transpose() * cov.inverse() * kstar)(0, 0)) * y_std;
                trlen = b.cwiseProduct(y_std) * epsilon + beta.cwiseProduct(y_std).cwiseProduct(stds) + etax_v;

                xvecnext_lout = xvec + means - trlen - ellin;
                xvecnext_uout = xvec + means + trlen + ellin;

                xvecnext_lin = xvec + means - trlen;
                xvecnext_uin = xvec + means + trlen;

                int qflag = 0;
                if ((xrange_l.array() <= xvecnext_lout.array()).all() == 1 && (xvecnext_uout.array() <= xrange_u.array()).all() == 1)
                {
                    Qind_lout = (xvecnext_lout - xrange_l).cwiseQuotient((2 / pow(3, 0.5)) * etax_v).array() + 1;
                    Qind_uout = (xvecnext_uout - xrange_l).cwiseQuotient((2 / pow(3, 0.5)) * etax_v);

                    Qind_lin = (xvecnext_lin - xrange_l).cwiseQuotient((2 / pow(3, 0.5)) * etax_v).array() + 1;
                    Qind_uin = (xvecnext_uin - xrange_l).cwiseQuotient((2 / pow(3, 0.5)) * etax_v);


                    qflag = safeF(Qsafe, Qind_lout, Qind_uout, Qind_lin, Qind_uin, Xqlist[0], Xqlist[1], Xqlist[2], alpha, Lambdax, gamma);
                }
                if (qflag == 1)
                {
                    cout << idq << "/" << Qind.rows() << ", [" << xvec(0, 0) << "," << xvec(1, 0) << "," << xvec(2, 0) << "]" << endl;
                    Cs(idq, 0) = Uq(idu, 0);
                    Cs(idq, 1) = Uq(idu, 1);;
                    break;
                }
                if (uflag == 0)
                {
                    Q[Qind(idq, 0)][Qind(idq, 1)][Qind(idq, 2)] = 0;
                }
            }
        }
    }else if (flag_refcon == 1){
        for (int idq = 0; idq < Qind.rows(); idq++)
        {
            double x_norm_min = 10000;

            for (int idu = 0; idu < Uq.rows(); idu++)
            {
                xvec << Xqlist[0](Qind(idq, 0), 0), Xqlist[1](Qind(idq, 1), 0), Xqlist[2](Qind(idq, 2), 0);
                zvec << Xqlist[0](Qind(idq, 0), 0), Xqlist[1](Qind(idq, 1), 0), Xqlist[2](Qind(idq, 2), 0), Uq(idu, 0), Uq(idu, 1);

                kstar = kstarF(alpha, Lambda, zvec, ZT);
                means = (kstar.transpose() * xi).transpose();
                xvecnext = xvec + means;
                if (xvecnext(2, 0) > 2 * 3.14159265358592){
                    xvecnext(2, 0) = xvecnext(2, 0) / 2 * 3.14159265358592;
                }

                if (x_norm_min > xvecnext.norm()){
                    x_norm_min = xvecnext.norm();
                    Cs(idq, 0) = Uq(idu, 0);
                    Cs(idq, 1) = Uq(idu, 1);
                }
            }
            cout << idq << "/" << Qind.rows() << endl;
        }
    }
    return forward_as_tuple(Q, Cs);
}

PYBIND11_MODULE(safetygame, m)
{
    m.doc() = "my test module";
    m.def("operation", &operation<double>, "");
}

// g++ -O3 -Wall -shared -std=c++14 -undefined dynamic_lookup safetygame.cpp -o safetygame$(python3-config --extension-suffix)
