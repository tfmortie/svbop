/* 
    Author: Thomas Mortier 2019-2020

    Header flat model
*/

#ifndef FLAT_H
#define FLAT_H

#include <iostream>
#include <vector>
#include <string>
#include "model/model.h"

/* main class (flat) softmax model */
class FlatModel : Model
{
    private:
        Matrix W;
        Matrix D;
        double update(const feature_node* x, const unsigned long y, const double lr); /* forward & backward pass */
        void backward(const feature_node* x, const double lr); /* backward pass */
        std::string getWeightVector();
        void setWeightVector(std::string w_str);
        void free();

    public:
        FlatModel(const problem* prob);
        FlatModel(const char* model_file_name);
        ~FlatModel();

        void printStruct();
        void printInfo(const bool verbose = 0);
        void performCrossValidation(unsigned int k);
        void reset();
        void fit(const std::vector<unsigned long>& ign_index = {}, const bool verbose = 1);
        unsigned long predict(const feature_node* x);
        std::vector<double> predict_proba(const feature_node* x, const std::vector<unsigned long> yv = {});
        unsigned long getNrClass();
        unsigned long getNrFeatures();
        void save(const char* model_file_name);
        void load(const char* model_file_name);
};

#endif