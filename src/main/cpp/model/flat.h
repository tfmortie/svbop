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
#include "Eigen/Dense"
#include "Eigen/SparseCore"

/* main class (flat) softmax model */
class FlatModel : Model
{
    private:
        Eigen::MatrixXd W;
        Eigen::MatrixXd D;
        double update(const Eigen::SparseVector<double>& x, const unsigned long y, const double lr); /* forward & backward pass */
        std::string getWeightVector();
        void setWeightVector(std::string w_str);

    public:
        FlatModel(problem* prob);
        FlatModel(const char* model_file_name, problem* prob);

        void printStruct();
        void printInfo(const bool verbose = 0);
        void performCrossValidation(unsigned int k);
        void reset();
        void fit(const std::vector<unsigned long>& ign_index = {}, const bool verbose = 1);
        unsigned long predict(const Eigen::SparseVector<double>& x);
        std::vector<double> predict_proba(const Eigen::SparseVector<double>& x, const std::vector<unsigned long> yv = {});
        std::vector<unsigned long> predict_ubop(const Eigen::SparseVector<double>& x);
        std::vector<unsigned long> predict_rbop(const Eigen::SparseVector<double>& x);
        unsigned long getNrClass();
        unsigned long getNrFeatures();
        void save(const char* model_file_name);
        void load(const char* model_file_name);
};

#endif