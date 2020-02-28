/* 
    Author: Thomas Mortier 2019-2020

    Some important math operations for models
*/

#include <cmath>
#include <assert.h>
#include <random>
#include <iostream>
#include <algorithm>
#include "mmath.h"
#include "Eigen/Dense"
#include "Eigen/SparseCore"

/*
    Stochastic gradient descent.
    W = W - lr*D

    Argument:
        ind: standard update if ind <0, fast update otherwise 
*/
void sgd(Eigen::MatrixXd& W, const Eigen::MatrixXd& D, const double lr, const long ind)
{
    // fast learning?
    if (ind < 0)
        W = W - lr*D;
    else
        W.col(ind) = W.col(ind) - lr*D.col(ind);
}

/*
    Adaptive moment estimation (Adam).
    See https://arxiv.org/pdf/1412.6980.pdf

    Argument:
        ind: standard update if ind <0, fast update otherwise 
*/
void adam(Eigen::MatrixXd& W, const Eigen::MatrixXd& D, Eigen::MatrixXd& M, Eigen::MatrixXd& V, const double lr, const unsigned long t, const long ind)
{
    if (ind < 0)
    {
        M = B_1*M + (1-B_1)*D; /* updated biased first moment estimate */
        V = B_2*V.array() + (1-B_2)*D.array().pow(2); /* updated biased second raw moment estimate */
        Eigen::MatrixXd M_H {M/(1-std::pow(B_1,t))}; /* compute bias-corrected first moment estimate */
        Eigen::MatrixXd V_H {V/(1-std::pow(B_2,t))}; /* compute bias-corrected second raw moment estimate */
        // and update parameters
        V_H = V_H.array().sqrt()+E;
        M_H = M_H.cwiseQuotient(V_H);
        M_H = lr * M_H;
        W = W - M_H;
    }
    else
    {
        M.col(ind) = B_1*M.col(ind) + (1-B_1)*D.col(ind); /* updated biased first moment estimate */
        V.col(ind) = B_2*V.array().col(ind) + (1-B_2)*D.col(ind).array().pow(2); /* updated biased second raw moment estimate */
        Eigen::VectorXd M_H {M.col(ind)/(1-std::pow(B_1,t))}; /* compute bias-corrected first moment estimate */
        Eigen::VectorXd V_H {V.col(ind)/(1-std::pow(B_2,t))}; /* compute bias-corrected second raw moment estimate */
        // and update parameters
        V_H = V_H.array().sqrt()+E;
        M_H = M_H.cwiseQuotient(V_H);
        M_H = lr * M_H;
        W.col(ind) = W.col(ind) - M_H;
    }
}

/*
    Stable softmax
    o = exp(o-max(o))/sum(exp(o-max(o)))
*/
void softmax(Eigen::VectorXd& o)
{
    // get max
    double x_max = o.maxCoeff();
    // subtract with max and exponentiate (making softmax stable)
    o = o.array()-x_max;
    // exponentiate and divide by its sum
    o = o.array().exp();
    o = o/o.sum();
}

/* 
    Set derivatives of softmax wrt categorical cross-entropy loss
    forall k: D[:, k] = (ok-tk)*x
    where for k=i tk=1 else tk=0
*/
void dvscalm(Eigen::MatrixXd& D, const Eigen::VectorXd& o, const unsigned long i, const Eigen::SparseVector<double>& x)
{
    for (unsigned long k=0; k<D.cols(); ++k)
    {
        // ok-tk
        double diff {(o[k]-static_cast<double>((k==i ? 1 : 0)))};
        // (ok-tk)*x
        D.col(k) = x*diff;
    }
}

/*
    Initialize weights in D from uniform distribution U(min,max).
*/
void inituw(Eigen::MatrixXd& W, const double min, const double max)
{
    double range {max-min};
    W = Eigen::MatrixXd::Random(W.rows(), W.cols());
    W = (W + Eigen::MatrixXd::Constant(W.rows(), W.cols(), 1.))*range/2.;
}