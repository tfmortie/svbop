/* 
    Author: Thomas Mortier 2019-2020

    Header math operations for models
*/

#ifndef MATH_H
#define MATH_H

#include "model/model.h"
#include "Eigen/Dense"
#include "Eigen/SparseCore"

const double EPS = 0.001; /* min. cut-off for probability (avoid log underflow) */ 
/* some constants for Adam optimizer */
const double B_1 = 0.9;
const double B_2 = 0.999;
const double E = 0.00000001;

void sgd(Eigen::MatrixXd& W, const Eigen::MatrixXd& D, const double lr, const long ind = -1);
void adam(Eigen::MatrixXd& W, const Eigen::MatrixXd& D, Eigen::MatrixXd& M, Eigen::MatrixXd& V, const double lr, const unsigned long t, const long ind = -1);

void softmax(Eigen::VectorXd& o);
void dvscalm(Eigen::MatrixXd& D, const Eigen::VectorXd& o, const unsigned long i, const Eigen::SparseVector<double>& x);
void inituw(Eigen::MatrixXd& W, const double min, const double max);

#endif