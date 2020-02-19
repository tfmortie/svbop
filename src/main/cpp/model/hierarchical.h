/* 
    Author: Thomas Mortier 2019-2020

    Header hierarchical softmax model
*/

#ifndef HIER_H
#define HIER_H

#include <iostream>
#include <vector>
#include <string>
#include "model/model.h"
#include "Eigen/Dense"
#include "Eigen/SparseCore"

/* class which represents a node in the hierarchical softmax model */
class HNode
{
    public: 
        Eigen::MatrixXd W;
        Eigen::MatrixXd D;

        HNode(const problem& prob); /* will be called on root */
        HNode(std::vector<unsigned long> y, const problem& prob);
        
        std::vector<unsigned long> y;
        std::vector<HNode*> chn;
        unsigned long predict(const Eigen::SparseVector<double>& x); /* predict child/branch */
        double predict(const Eigen::SparseVector<double>& x, const unsigned long ind); /* get branch probability of child node with index ind */
        double update(const Eigen::SparseVector<double>& x, const unsigned long ind, const double lr, const bool fast = 0); /* forward & backward pass */
        void reset();
        void addChildNode(std::vector<unsigned long> y, const problem& p);     
        std::string getWeightVector();
        void setWeightVector(std::string w_str);
        void print();
};

/* main class hierarchical softmax model */
class HierModel : Model
{
    private:
        HNode* root;
        
    public:
        HierModel(const problem* prob);
        HierModel(const char* model_file_name);
        ~HierModel();

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