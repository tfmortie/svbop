/*
Author: Thomas Mortier 2019

Implementation of hierarchical model 
*/

// TODO: finalize comments

#include "model/hierarchical.h"
#include <iostream>
#include <utility> 
#include <algorithm>
#include <sstream>
#include <queue>

HNode::HNode(const problem &prob) : p{prob}
{
    // first init W matrix
    this->w = W_hnode{new double*[static_cast<unsigned long>(p.n)], static_cast<unsigned long>(p.n), 0};
    // set y attribute of this node (i.e., root)
    this->y = this->p.h_struct[0];
    // now construct tree
    for (unsigned int i = 1; i < this->p.h_struct.size(); ++i)
        this->addChildNode(this->p.h_struct[i], this->p);    
}   

HNode::HNode(std::vector<int> y, const problem &prob) : y{y}, p{prob}
{
    // only init W if internal node!
    if (y.size() > 1)
        this->w = W_hnode{new double*[static_cast<unsigned long>(prob.n)], static_cast<unsigned long>(prob.n), 0};
} 

HNode::~HNode()
{
    this->free();
}

void HNode::addChildNode(std::vector<int> y, const problem &p)
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
            this->chn[static_cast<unsigned long>(ind)]->addChildNode(y,p);
        else
        {
            // no children for which y is a subset, hence, put in children list
            HNode* new_node = new HNode{y, p};
            this->chn.push_back(new_node);
            unsigned long tot_len_y_chn {0};
            for (auto c : this->chn)
                tot_len_y_chn += c->y.size();
            // check if the current node has all its children
            if (tot_len_y_chn == this->y.size())
            {
                // allocate weight vector 
                for (unsigned int i=0; i < static_cast<unsigned int>(this->p.n); ++i)
                    this->w.value[i] = new double[this->chn.size()];
            }
            // set k size attribute
            this->w.k = this->chn.size();
        }
    }
    else
    { 
        // no children yet, hence, put in children list
        HNode* new_node = new HNode{y, p};
        this->chn.push_back(new_node);
    }
}

void HNode::free()
{
    if (this->y.size() > 1)
    { 
        for (unsigned int i = 0; i < static_cast<unsigned int>(this->p.n); ++i)
            delete[] this->w.value[i];

        delete[] this->w.value;
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
        std::cout << "NODE(" << oss.str() << ")[";
        for (auto c : this->chn)
            c->print();
        std::cout << "]";
    } 
    else
        std::cout << "NODE(" << oss.str() << ")";
}

HierModel::HierModel(const problem &prob) 
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

void HierModel::printStructure()
{
    std::cout << "[info] Structure tree: \n";
    if (root != nullptr)
        this->root->print();
    std::cout << "\n";
}

void HierModel::printInfo()
{
    std::cout << "---------------------------------------------------\n";
    std::cout << "[info] Hierarchical model: \n";
    std::cout << "---------------------------------------------------\n";
    std::cout << "  * Number of features              = " << this->root->p.n << '\n';
    std::cout << "  * Number of samples               = " << this->root->p.l << '\n';
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