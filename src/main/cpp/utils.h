/* 
    Author: Thomas Mortier 2019-2020

    Header utils  
*/

#ifndef UTILS_H
#define UTILS_H

#include "model/model.h"
#include "Eigen/Dense"
#include "Eigen/SparseCore"

const double EPS = 0.001; /* min. cut-off for probability (avoid log underflow) */ 

void softmax(Eigen::VectorXd& o);
void dvscalm(Eigen::MatrixXd& D, const Eigen::VectorXd& o, const unsigned long i, const Eigen::SparseVector<double>& x);

#endif