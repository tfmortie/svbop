/*
Author: Thomas Mortier 2019

Header flat model
*/

// TODO: finalize comments

#ifndef FLAT_H
#define FLAT_H

#include "liblinear/linear.h"
#include <iostream>

class FlatModel 
{
    private:
        const problem* prob;
        const parameter* param;
        model* model;
        int* class_to_label_dict;
        void free();

    public:
        FlatModel(const problem* prob, const parameter* param) : prob{prob}, param{param}, model{nullptr} {};
        FlatModel(const char* model_file_name);
        ~FlatModel();

        void printInfo();
        void performCrossValidation();
        void fit();
        double predict(const feature_node* x);
        void predict_proba(const feature_node* x, double* prob_estimates);
        void checkParam();
        int getNrClass();
        void save(const char* model_file_name);
};

#endif