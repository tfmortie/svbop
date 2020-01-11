/*
Author: Thomas Mortier 2019

Header hierarchical model
*/

// TODO: finalize comments

#ifndef HIER_H
#define HIER_H

#include "model/model.h"
#include <iostream>
#include <vector>
#include <string>

struct W_hnode
{
    double** value; /* should be D x K */
    unsigned long d; /* D */
    unsigned long k; /* K */
};

/*
Class which represents a node in HierModel
*/
class HNode
{
    private:
        W_hnode w;
        void free();   

    public: 
        HNode(const problem &prob); /* will be called on root */
        HNode(std::vector<int> y, const problem &prob);
        ~HNode();
        std::vector<int> y;
        std::vector<HNode*> chn;
        const problem &p;
        void addChildNode(std::vector<int> y, const problem &p);     
        void print();
};

class HierModel
{
    private:
        HNode* root;
        
    public:
        HierModel(const problem &prob);
        ~HierModel();

        void printStructure();
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