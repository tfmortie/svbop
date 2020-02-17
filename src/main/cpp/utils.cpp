/* 
    Author: Thomas Mortier 2019-2020

    Some important math operations 
*/

#include "utils.h"
#include <cmath>
#include <assert.h>
#include <random>
#include <iostream>
#include <algorithm>

/* y = alpha*W.Tx (jagged + sparse array) */
void dgemv(const double alpha, const double** W, const feature_node* x, double* y, const unsigned long k)
{
    for (unsigned long j=0; j<k; ++j)
    {
        unsigned long ind {0};
        while (x[ind].index != -1)
        {
            y[j] += alpha*x[ind].value*W[x[ind].index-1][j];
            ++ind;
        }
    }
}

/* y = alpha*W.Tx (nonjagged row-major order + sparse array) */
void dgemv(const double alpha, const double* W, const feature_node* x, double* y, const unsigned long k)
{
    for (unsigned long j=0; j<k; ++j)
    {
        unsigned long ind {0};
        while (x[ind].index != -1)
        {
            y[j] += alpha*x[ind].value*W[(static_cast<unsigned long>(x[ind].index-1)*k)+j];
            ++ind;
        }
    }
}

/* y = alpha*W.Tx (jagged + dense array)*/
void dgemv(const double alpha, const double** W, const double* x, double* y, const unsigned long d, const unsigned long k)
{
    for (unsigned long j=0; j<k; ++j)
    {
        for(unsigned long i=0; i<d; ++i)
            y[j] += alpha*x[i]*W[i][j];
    }
}

/* y = alpha*W.Tx (nonjagged row-major order + dense array) */
void dgemv(const double alpha, const double* W, const double* x, double* y, const unsigned long d, const unsigned long k)
{
    for (unsigned long j=0; j<k; ++j)
    {
        for (unsigned long i=0; i<d; ++i)
            y[j] = alpha*x[i]*W[(i*k)+j];
    }
}

/* D[:,i] = alpha*x (jagged + sparse array) */
void dvscalm(const double alpha, const feature_node* x, double** D, const unsigned long k, const unsigned long i)
{
    assert(i<k);
    unsigned long ind {0};
    while (x[ind].index != -1)
    {
        D[x[ind].index-1][i] = alpha*x[ind].value;
        ++ind;
    }
}

/* D[:,i] = alpha*x (nonjagged row-major order + sparse array) */
void dvscalm(const double alpha, const feature_node* x, double* D, const unsigned long k, const unsigned long i)
{
    assert(i<k);
    unsigned long ind {0};
    while (x[ind].index != -1)
    {
        D[(static_cast<unsigned long>(x[ind].index-1)*k)+i] = alpha*x[ind].value;
        ++ind;
    }
}

/* D[:,i] = alpha*x (jagged + dense array) */
void dvscalm(const double alpha, const double* x, double** D, const unsigned long d, const unsigned long k, const unsigned long i)
{
    assert(i<k);
    for(unsigned long n=0; n<d; ++n)
        D[n][i] = alpha*x[n];
}

/* D[:,i] = alpha*x (nonjagged row-major order + dense array) */
void dvscalm(const double alpha, const double* x, double* D, const unsigned long d, const unsigned long k, const unsigned long i)
{
    assert(i<k);
    for(unsigned long n=0; n<d; ++n)
        D[(n*k)+i] = alpha*x[n];
}

/* x = alpha*x */
void dvscal(const double alpha, double* x, const unsigned long d)
{
    for (unsigned long i=0; i<d; ++i)
        x[i] = x[i]*alpha;
}

/* W[:][i] = W[:][i]-alpha*D[:][i] (jagged array) */
void dsubmv(const double alpha, double** W, const double** D, const unsigned long d, const unsigned long k, const unsigned long i)
{
    assert(i<k);
    for (unsigned long n=0; n<d; ++n)
        W[n][i] = W[n][i]-(alpha*D[n][i]);
}

/* W[:][i] = W[:][i]-alpha*D[:][i] (nonjagged row-major order array) */
void dsubmv(const double alpha, double* W, const double* D, const unsigned long d, const unsigned long k, const unsigned long i)
{
    assert(i<k);
    for (unsigned long n=0; n<d;++n)
        W[(n*k)+i] = W[(n*k)+i]-alpha*(D[(n*k)+i]);
}

/* x = exp(x)/sum(exp(x)) */
void softmax(double* x, const unsigned long k)
{
    // first calculate max 
    double x_max {*std::max_element(x, x+k)};
    // exp and calculate Z 
    double Z {0.0};
    for (unsigned long i=0; i<k; ++i)
    {
        Z += std::exp(x[i]-x_max);
        x[i] = std::exp(x[i]-x_max);
    }
    // divide x by denum
    dvscal(1/Z, x, k);
}

/* function which init. W (jagged) with values from uniform distribution U(min, max) */
void initUW(const double min, const double max, double** W, const unsigned long d, const unsigned long k)
{
    // create rng
    std::random_device rd; /* get seed for the rn engine */
    std::mt19937 gen(rd()); /* mersenne_twister_engine seeded with rd() */
    std::uniform_real_distribution<> dis(min, max);
    // init W
    for (unsigned long i=0; i<d; ++i)
    {
        for (unsigned long j=0; j<k; ++j)
            W[i][j] = dis(gen);
    }
}

/* function which init. W (nonjagged row-major order) with values from uniform distribution U(min, max) */
void initUW(const double min, const double max, double* W, const unsigned long d, const unsigned long k)
{
    // create rng
    std::random_device rd; /* get seed for the rn engine */
    std::mt19937 gen(rd()); /* mersenne_twister_engine seeded with rd() */
    std::uniform_real_distribution<> dis(min, max);
    // init W
    for (unsigned long i=0; i<(d*k); ++i)
        W[i] = dis(gen);
}

/* transforms feature_node array to double array */
double* ftvToArr(const feature_node *x, const unsigned long size)
{
    double* arr = new double[size]();
    unsigned long i = 0;
    while(x[i].index != -1)
    {
        arr[x[i].index-1] = x[i].value; 
        ++i;
    }
    return arr;
}
