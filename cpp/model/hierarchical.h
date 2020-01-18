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

/* Weight matrix */
struct W_hnode
{
    double** value; /* should be D x K */
    unsigned long d; /* D */
    unsigned long k; /* K */
};

/* Delta for backward pass */
struct d_hnode
{
    double* value;
    unsigned long d; /* D */
    long ind; /* index to which delta applies */
};

/*
Class which represents a node in HierModel
*/
class HNode
{
    private:
        W_hnode W;
        d_hnode D;
        void free();   

    public: 
        HNode(const problem &prob); /* will be called on root */
        HNode(std::vector<int> y, const problem &prob);
        ~HNode();

        std::vector<int> y;
        std::vector<HNode*> chn;
        double update(const feature_node *x, const long ind, const float lr); /* forward & backward pass */
        void backward(const feature_node *x, const float lr); /* backward pass */
        void addChildNode(std::vector<int> y, const problem &p);     
        void print();
};

class HierModel
{
    private:
        HNode* root;
        const problem &prob;
        
    public:
        HierModel(const problem &prob);
        ~HierModel();

        void printStruct();
        void print();
        void printInfo();
        void performCrossValidation();
        void fit(const float lr);
        double predict(const feature_node *x);
        void predict_proba(const feature_node* x, double* prob_estimates);
        void checkParam();
        int getNrClass();
        void save(const char* model_file_name);
};

#endif