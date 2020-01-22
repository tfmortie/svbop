/*
Author: Thomas Mortier 2019

Header hierarchical model
*/

// TODO: finalize comments

#ifndef HIER_H
#define HIER_H

#include <iostream>
#include <vector>
#include <string>
#include "liblinear/linear.h"

/* Weight matrix */
struct W_hnode
{
    double** value; /* should be D x K */
    unsigned long d; /* D */
    unsigned long k; /* K */
};

/* Delta matrix for backward pass */
struct D_hnode
{
    double** value;
    unsigned long d; /* D */
    unsigned long k; /* K */
};

/*
Class which represents a node in HierModel
*/
class HNode
{
    private:
        W_hnode W;
        D_hnode D;
        void free();   

    public: 
        HNode(const problem& prob); /* will be called on root */
        HNode(std::vector<int> y, const problem& prob);
        ~HNode();

        std::vector<int> y;
        std::vector<HNode*> chn;
        unsigned int predict(const feature_node* x); /* predict child/branch */
        double update(const feature_node* x, const long ind, const double lr); /* forward & backward pass */
        void backward(const feature_node* x, const double lr); /* backward pass */
        void reset();
        void addChildNode(std::vector<int> y, const problem& p);     
        void print();
};

class HierModel
{
    private:
        const problem* prob;
        const parameter* param;
        HNode* root;
        
    public:
        HierModel(const problem* prob, const parameter* param);
        ~HierModel();

        void printStruct();
        void print();
        void printInfo(const bool verbose = 0);
        void performCrossValidation(unsigned int k);
        void reset();
        void fit(const std::vector<unsigned int>& ign_index = {}, const bool verbose = 1);
        double predict(const feature_node* x);
        int getNrClass();
        void save(const char* model_file_name);
        void load(const char* model_file_name);
};

#endif