/* 
    Author: Thomas Mortier 2019-2020

    Some important math operations 
*/

#include <cmath>
#include <assert.h>
#include <random>
#include <iostream>
#include <algorithm>
#include "utils.h"
#include "Eigen/Dense"
#include "Eigen/SparseCore"

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