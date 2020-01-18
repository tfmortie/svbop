/*
Author: Thomas Mortier 2019

Header utils  
*/

#ifndef UTILS_H
#define UTILS_H

#include "liblinear/linear.h"

void dgemv(const double alpha, const double** W, const double* x, double* y, const unsigned long d, const unsigned long k);
void dscal(const double alpha, double* x, const unsigned long d);
void dsubmv(const double alpha, double** W, const double* x, const unsigned long d, const unsigned long k, const unsigned long i);
void softmax(double* x, const unsigned long d);
double* ftvToArr(const feature_node *x, const unsigned long size);

#endif