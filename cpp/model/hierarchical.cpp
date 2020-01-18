/*
Author: Thomas Mortier 2019

Implementation of hierarchical model 
*/

// TODO: finalize comments
// TODO: initializers!!!!

#include "model/hierarchical.h"
#include "utils.h"
#include <iostream>
#include <utility> 
#include <algorithm>
#include <sstream>
#include <queue>

HNode::HNode(const problem &prob) 
{
    // first init W matrix
    this->W = W_hnode{new double*[static_cast<unsigned long>(prob.n)], static_cast<unsigned long>(prob.n), 0};
    // init D vector
    this->D = d_hnode{new double[static_cast<unsigned long>(prob.n)], static_cast<unsigned long>(prob.n), -1};
    // set y attribute of this node (i.e., root)
    this->y = prob.h_struct[0];
    // now construct tree
    for (unsigned int i = 1; i < prob.h_struct.size(); ++i)
        this->addChildNode(prob.h_struct[i], prob);    
}   

HNode::HNode(std::vector<int> y, const problem &prob) : y{y}
{
    // only init D if internal node!
    if (y.size() > 1)
    {
        this->W = W_hnode{new double*[static_cast<unsigned long>(prob.n)], static_cast<unsigned long>(prob.n), 0};
        this->D = d_hnode{new double[static_cast<unsigned long>(prob.n)], static_cast<unsigned long>(prob.n), -1};
    }
} 

HNode::~HNode()
{
    this->free();
}

double HNode::forward(const feature_node *x, const long ind)
{
    double* o {new double[this->W.k]}; // array of exp
    // convert feature_node arr to double arr
    double* x_arr {ftvToArr(x, this->W.d)}; 
    // Wtx
    dgemv(1.0, const_cast<const double **>(this->W.value), x_arr, o, this->W.d, this->W.k);
    // apply softmax
    softmax(o, this->W.k); 
    // set grad loc and delta's 
    this->D.ind = ind;
    dscal((o[ind]-1), this->D.value, this->D.d);
    double p {o[ind]};
    // delete
    delete[] x_arr;
    delete[] o;
    return p;
}

void HNode::backward(const feature_node *x, const float lr)
{
    if (this->D.ind != -1)
    {    

        dsubmv(lr, this->W.value, this->D.value, this->D.d, this->W.k, static_cast<unsigned long>(this->D.ind));
        // reset gradient
        this->D.ind = -1;
    }
    else
    {
        std::cerr << "[error] Backward operation without preceding forward pass!\n";
        exit(1);
    }
}

void HNode::addChildNode(std::vector<int> y, const problem &prob)
{
    // todo: optimize?
    // check if leaf or internal node 
    if (this->chn.size() > 0)
    {
        // check if y is a subset of one of the children
        int ind = -1;
        for (unsigned int i = 0; i < this->chn.size(); ++i)
        {
            if (std::includes(this->chn[i]->y.begin(), this->chn[i]->y.end(), y.begin(), y.end()) == 1)
            {
                ind = static_cast<int>(i);
                break;
            }
        }
        if (ind != -1)
            // subset found, hence, recursively pass to child
            this->chn[static_cast<unsigned long>(ind)]->addChildNode(y, prob);
        else
        {
            // no children for which y is a subset, hence, put in children list
            HNode* new_node = new HNode{y, prob};
            this->chn.push_back(new_node);
            unsigned long tot_len_y_chn {0};
            for (auto c : this->chn)
                tot_len_y_chn += c->y.size();
            // check if the current node has all its children
            if (tot_len_y_chn == this->y.size())
            {
                // allocate weight vector 
                for (unsigned int i=0; i < static_cast<unsigned int>(prob.n); ++i)
                    this->W.value[i] = new double[this->chn.size()];
            }
            // set k size attribute
            this->W.k = this->chn.size();
        }
    }
    else
    { 
        // no children yet, hence, put in children list
        HNode* new_node = new HNode{y, prob};
        this->chn.push_back(new_node);
    }
}

void HNode::free()
{
    if (this->y.size() > 1)
    { 
        for (unsigned int i = 0; i < static_cast<unsigned int>(this->W.d); ++i)
            delete[] this->W.value[i];

        delete[] this->W.value;
        delete[] this->D.value;
    }
}  

void HNode::print()
{
    std::ostringstream oss;
    if(!this->y.empty())
    {
        // Convert all but the last element to avoid a trailing ","
        std::copy(this->y.begin(), this->y.end()-1,
        std::ostream_iterator<int>(oss, ","));

        // Now add the last element with no delimiter
        oss << this->y.back();
    }
    if (this->chn.size() != 0)
    {
        std::cout << "NODE(" << oss.str() << ")\n";
        std::cout << "[\n" ;
        for (auto c : this->chn)
            c->print();
        std::cout << "]\n";
    } 
    else
        std::cout << "NODE(" << oss.str() << ")\n";
}

HierModel::HierModel(const problem &prob) : prob{prob}
{
    // construct tree 
    root = new HNode(prob);
}

HierModel::~HierModel()
{
    if (root != nullptr)
    {
        std::queue<HNode*> visit_list; 
        visit_list.push(this->root);
        while(!visit_list.empty())
        {
            HNode* visit_node = visit_list.front();
            visit_list.pop();
            if (!visit_node->chn.empty())
            {
                for(auto* c : visit_node->chn)
                    visit_list.push(c);
            }
            // free node
            delete visit_node;
        }
    }
}

void HierModel::printStruct()
{
    std::cout << "[info] Structure:\n";
    for (auto &v : this->prob.h_struct)
    {
        std::cout << "[";
        for (auto &el: v)
        {
            std::cout << el << ",";
        }
        std::cout << "]\n";
    }
}

void HierModel::print()
{
    if (root != nullptr)
        this->root->print();
}

void HierModel::printInfo()
{
    std::cout << "---------------------------------------------------\n";
    std::cout << "[info] Hierarchical model: \n";
    std::cout << "---------------------------------------------------\n";
    std::cout << "  * Number of features              = " << this->prob.n << '\n';
    std::cout << "  * Number of samples               = " << this->prob.l << '\n';
    std::cout << "  * Model =\n";
    this->print();
    std::cout << "---------------------------------------------------\n\n";
}

void HierModel::performCrossValidation()
{
    //TODO: implement!

}

void HierModel::fit()
{
    //TODO: implement!
}

// predict class with highest probability 
double HierModel::predict(const feature_node *x)
{
    if (root == nullptr)
    {
        std::cerr << "[warning] Model has not been fitted yet";
        return -1.0;
    }
    else
    {
        // TODO: implement!
        return -1.0;
    }
}

void HierModel::predict_proba(const feature_node* x, double* prob_estimates)
{
    if (root == nullptr)
    {
        std::cerr << "[warning] Model has not been fitted yet!";
    }
    else
    {
        // fill in probability estimates vector 
        // TODO: implement!
    }
}
    
// check problem and parameter before training
void HierModel::checkParam()
{
    // TODO: implement!
}

// get number of classes of fitted model
int HierModel::getNrClass()
{
    if (root == nullptr)
    {
        std::cerr << "[warning] Model has not been fitted yet!\n";
        return 0;
    }
    else
        return this->root->y.size();
}

// save model
void HierModel::save(const char* model_file_name)
{
    // TODO: implement!
}