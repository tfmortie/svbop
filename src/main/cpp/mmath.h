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

void softmax(Eigen::VectorXd& o);
void dvscalm(Eigen::MatrixXd& D, const Eigen::VectorXd& o, const unsigned long i, const Eigen::SparseVector<double>& x);
void inituw(Eigen::MatrixXd& W, const double min, const double max);

#endif