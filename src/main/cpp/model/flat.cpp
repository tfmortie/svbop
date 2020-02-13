/* Author: Thomas Mortier 2019-2020

   Implementation of model with softmax

   TODO: comments
   TODO: optimize (allow sparse features (feature_node))!
*/

#include "model/flat.h"
#include "data.h"
#include "utils.h"

/* CONSTRUCTOR AND DESTRUCTOR */

FlatModel::FlatModel(const problem* prob) : Model(prob)
{
    // first init W matrix
    this->W = Matrix{new double*[prob->n], prob->n, prob->hstruct[0].size()};
    // init D vector
    this->D = Matrix{new double*[prob->n], prob->n, 0};
    // init W
    initUW(static_cast<double>(-1.0/this->W.d), static_cast<double>(1.0/this->W.d), this->W.value, this->W.d, this->W.k);
}

FlatModel::~FlatModel()
{
    for (unsigned long i = 0; i < this->W.d; ++i)
    {
        delete[] this->W.value[i];
        delete[] this->D.value[i];
    }
    delete[] this->W.value;
    delete[] this->D.value;
}

/* PRIVATE */

double FlatModel::update(const feature_node* x, const double lr)
{
    // forward step
    double* o {new double[this->W.k]()}; // array of exp
    // convert feature_node arr to double arr
    double* x_arr {ftvToArr(x, this->W.d)}; 
    // Wtx
    dgemv(1.0, const_cast<const double**>(this->W.value), x_arr, o, this->W.d, this->W.k);
    // apply softmax
    softmax(o, this->W.k); 
    // set delta's 
    double t {0.0};
    for(unsigned long i=0; i<this->D.k; ++i)
    {
        if(static_cast<const long&>(i) == ind)
            t = 1.0;
        else
            t = 0.0;

        dvscalm((o[i]-t), x_arr, this->D.value, this->D.d, this->D.k, i);
    }
    // backward step
    this->backward(x, lr);
    double p {o[ind]};
    // delete
    delete[] x_arr;
    delete[] o;
    return p;
}
