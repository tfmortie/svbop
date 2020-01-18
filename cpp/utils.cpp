/*
Author: Thomas Mortier 2019

Some important math operations 
*/

#include "utils.h"
#include "liblinear/linear.h"
#include <cmath>
#include <assert.h>
#include <random>

// TODO: in future perhaps use LAPACK/BLAS for matrix/vector multiplications...

/*
y = alpha*W.Tx
*/
void dgemv(const double alpha, const double** W, const double* x, double* y, const unsigned long d, const unsigned long k)
{
    for (unsigned long j=0; j<k; ++j)
    {
        for(unsigned long i=0; i<d; ++i)
            y[j] += alpha*x[i]*W[i][j];
    }
}

/*
x = alpha*x
*/
void dscal(const double alpha, double* x, const unsigned long d)
{
    for (unsigned long i=0; i<d; ++i)
        x[i] = x[i]*alpha;
}

/*
W[:][i] = W[:][i]-alpha*x
*/
void dsubmv(const double alpha, double** W, const double* x, const unsigned long d, const unsigned long k, const unsigned long i)
{
    assert(i<k);
    for(unsigned long n=0; n<d; ++n)
        W[n][i] = W[n][i]-(alpha*x[i]);
}

/*
x = exp(x)/sum(exp(x))
*/
void softmax(double* x, const unsigned long d)
{
    // first calculate Z 
    double Z {0.0};
    for (unsigned long i=0; i<d; ++i)
        Z += exp(x[i]);
    // divide x by denum
    dscal(1/Z, x, d);
}

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

/*
Transforms feature_node array to double array
*/
double* ftvToArr(const feature_node *x, const unsigned long size)
{
    double* arr = new double[size] {0};
    unsigned long i = 0;
    while(x[i].value != -1)
    {
        arr[x[i].index-1] = x[i].value;
        ++i;
    }
    return arr;
}
