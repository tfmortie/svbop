/*
Author: Thomas Mortier 2019

Header hierarchical model
*/

// TODO: finalize comments

#ifndef HIER_H
#define HIER_H

#include "model/model.h"
#include <iostream>

class HierModel : public Model
{
    private:
        int* class_to_label_dict;
        void free();

    public:
        HierModel(const problem* prob, const parameter* param);
        HierModel(const char* model_file_name);
        ~HierModel();

        void printInfo();
        void performCrossValidation();
        void fit();
        double predict(const feature_node *x);
        void predict_proba(const feature_node* x, double* prob_estimates);
        void checkParam();
        int getNrClass();
        void save(const char* model_file_name);
};

#endif