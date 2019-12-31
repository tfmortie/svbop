/*
Author: Thomas Mortier 2019

Abstract class model
*/

// TODO: finalize comments
// TODO: decide on making Model class abstract...

#ifndef MODEL_U
#define MODEL_U

#include "liblinear/linear.h"

class Model
{
    protected:
        const problem* prob;
        const parameter* param;
        model* model;
        //void free();

    public:
        Model(const problem* prob, const parameter* param) : prob{prob}, param{param}, model{nullptr} {};
        Model(const char* model_file_name) : prob{nullptr}, param{nullptr}, model{load_model(model_file_name)} {};
        ~Model() {};
/*         virtual void printInfo() = 0;
        virtual void performCrossValidation() = 0;
        virtual void fit() = 0;
        virtual double predict(const feature_node *x) = 0;
        virtual void predict_proba(const feature_node *x, double* prob_estimates) = 0;
        virtual void checkParam() = 0;
        virtual int getNrClass() = 0;
        virtual void save(const char* model_file_name) = 0; */
};

#endif