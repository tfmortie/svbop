/*
Author: Thomas Mortier 2019

Implementation of hierarchical model 
*/

// TODO: finalize comments
// TODO: optimize (allow sparse features (feature_node))!
// TODO: change int type of y to unsigned long!
// TODO: change W double -> long double!

#include "model/hierarchical.h"
#include "data.h"
#include "utils.h"
#include <string>
#include <iostream>
#include <iomanip>
#include <utility> 
#include <algorithm>
#include <sstream>
#include <queue>
#include <cmath>
#include <iterator>
#include <vector>
#include <random>
#include <fstream>

HNode::HNode(const problem &prob) 
{
    // first init W matrix
    this->W = W_hnode{new double*[static_cast<unsigned long>(prob.n)], static_cast<unsigned long>(prob.n), 0};
    // init D vector
    this->D = D_hnode{new double*[static_cast<unsigned long>(prob.n)], static_cast<unsigned long>(prob.n), 0};
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
        // first init W matrix
        this->W = W_hnode{new double*[static_cast<unsigned long>(prob.n)], static_cast<unsigned long>(prob.n), 0};
        // init D vector
        this->D = D_hnode{new double*[static_cast<unsigned long>(prob.n)], static_cast<unsigned long>(prob.n), 0};
    }
} 

HNode::~HNode()
{
    this->free();
}

unsigned int HNode::predict(const feature_node *x)
{
    // forward step
    double* o {new double[this->W.k]()}; // array of exp
    // convert feature_node arr to double arr
    double* x_arr {ftvToArr(x, this->W.d)}; 
    // Wtx
    dgemv(1.0, const_cast<const double**>(this->W.value), x_arr, o, this->W.d, this->W.k);
    // get index max
    double* max_o {std::max_element(o, o+this->W.k)}; 
    unsigned int max_i {static_cast<unsigned int>(static_cast<unsigned long>(max_o-o))};
    // delete
    delete[] x_arr;
    delete[] o;
    return max_i;
}

double HNode::update(const feature_node *x, const long ind, const double lr)
{
    // forward step
    double* o {new double[this->W.k]()}; // array of exp
    // convert feature_node arr to double arr
    double* x_arr {ftvToArr(x, this->W.d)}; 
    // Wtx
    dgemv(1.0, const_cast<const double**>(this->W.value), x_arr, o, this->W.d, this->W.k);
    // apply softmax
    softmax(o, this->W.k); 
    // set delta's 
    double t {0.0};
    for(unsigned long i=0; i<this->D.k; ++i)
    {
        if(static_cast<const long&>(i) == ind)
            t = 1.0;
        else
            t = 0.0;

        dvscalm((o[i]-t), x_arr, this->D.value, this->D.d, this->D.k, i);
    }
    // backward step
    this->backward(x, lr);
    double p {o[ind]};
    // delete
    delete[] x_arr;
    delete[] o;
    return p;
}

void HNode::backward(const feature_node *x, const double lr)
{
    for (unsigned long i=0; i<this->W.k; ++i)
        dsubmv(lr, this->W.value, const_cast<const double**>(this->D.value), this->W.d, this->W.k, i);
}

void HNode::reset()
{
    // reinitialize W
    initUW(static_cast<double>(-1.0/this->W.d), static_cast<double>(1.0/this->W.d), this->W.value, this->W.d, this->W.k);
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
                // allocate weight and delta vectors 
                for (unsigned int i=0; i<static_cast<unsigned int>(prob.n); ++i)
                {
                    this->W.value[i] = new double[this->chn.size()];
                    this->D.value[i] = new double[this->chn.size()]{0};
                }
            }
            // set k size attribute
            this->W.k = this->chn.size();
            this->D.k = this->chn.size();
            // init W
            initUW(static_cast<double>(-1.0/this->W.d), static_cast<double>(1.0/this->W.d), this->W.value, this->W.d, this->W.k);
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
        {
            delete[] this->W.value[i];
            delete[] this->D.value[i];
        }
        delete[] this->W.value;
        delete[] this->D.value;
    }
}  

unsigned long HNode::getNrFeatures()
{
    return this->W.d;
}

std::string HNode::getWeightVector()
{
    std::string ret_arr;
    // process all elements in row-major order
    for (unsigned long i=0; i<this->W.d; ++i)
    {
        for (unsigned long j=0; j<this->W.k; ++j)
        {

            std::stringstream str_stream;
            str_stream << std::fixed << std::setprecision(18) << std::to_string(this->W.value[i][j]);
            ret_arr += str_stream.str();
            if ((i!=this->W.d-1) || (j!=this->W.k-1))
                ret_arr += ' ';
        }
    }
    return ret_arr;
}

void HNode::setWeightVector(std::string w_str)
{
    // convert string to input stream
    std::istringstream istr_stream {w_str};
    // weights are separated by ' ', hence, split accordingly
    std::vector<std::string> tokens {std::istream_iterator<std::string> {istr_stream}, std::istream_iterator<std::string>{}};
    // run over weights in row-major order and save to W
    for (unsigned long i=0; i<this->W.d; ++i)
    {
        for (unsigned long j=0; j<this->W.k; ++j)
            this->W.value[i][j] = std::stod(tokens[(i*this->W.k)+j]);
    }
}

void HNode::print()
{
    std::ostringstream oss;
    if(!this->y.empty())
    {
        // convert all but the last element to avoid a trailing ","
        std::copy(this->y.begin(), this->y.end()-1,
        std::ostream_iterator<int>(oss, ","));

        // now add the last element with no delimiter
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

HierModel::HierModel(const problem* prob, const parameter* param) : prob{prob}, param{param}
{
    // construct tree 
    root = new HNode(*prob);
}

HierModel::HierModel(const char* model_file_name) : prob{nullptr}, param{nullptr}
{
    std::cout << "Loading model from " << model_file_name << "...\n";
    this->load(model_file_name);
    if (this->root == nullptr)
    {
        std::cerr << "[error] File "  << model_file_name << " does not exist!\n";
        exit(1);
    }
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
    if (root != nullptr)
        this->root->print();
}

void HierModel::printInfo(const bool verbose)
{
    if (root != nullptr)
    {
        std::cout << "---------------------------------------------------\n";
        std::cout << "[info] Hierarchical model: \n";
        std::cout << "---------------------------------------------------\n";
        std::cout << "  * Number of features              = " << this->root->getNrFeatures() << '\n';
        std::cout << "  * Number of classes               = " << this->root->y.size() << '\n';
        if (verbose)
        {
            std::cout << "  * Structure =\n";
            this->printStruct();
        }
        std::cout << "---------------------------------------------------\n\n";
    }
    else
    {
        std::cerr << "[warning] Model has not been constructed yet!\n";
    }
}

void HierModel::performCrossValidation(unsigned int k)
{
    if (this->root != nullptr && this->prob != nullptr)
    {
        std::cout << "---- " << k << "-Fold CV ----\n";
        // first create index vector
        std::vector<unsigned int> ind_arr;
        for(unsigned int i=0; i<static_cast<unsigned int>(this->prob->l); ++i)
        {
            ind_arr.push_back(i);
        }
        // now shuffle index vector 
        auto rng = std::default_random_engine {};
        std::shuffle(std::begin(ind_arr), std::end(ind_arr), rng);
        // calculate size of each test fold
        unsigned int ns_fold {static_cast<unsigned int>(static_cast<unsigned int>(this->prob->l)/k)};
        // start kfcv
        unsigned int iter {0};
        while(iter < k)
        {
            std::cout << "FOLD " << iter+1 << '\n';
            // first clear weights 
            this->reset();
            // extract test fold indices
            std::vector<unsigned int>::const_iterator i_start = ind_arr.begin() + iter*ns_fold;
            std::vector<unsigned int>::const_iterator i_stop = ind_arr.begin() + (iter+1)*ns_fold;
            std::vector<unsigned int> testfold_ind(i_start, i_stop);
            // now start fitting 
            this->fit(testfold_ind, 0);
            // and validate on training and test fold
            double acc {0.0};
            double n_cntr {0.0};
            for (unsigned int n = 0; n<static_cast<unsigned int>(this->prob->l); ++n)
            {
                if (std::find(testfold_ind.begin(), testfold_ind.end(), n) == testfold_ind.end())
                {
                    double pred {this->predict(this->prob->x[n])};
                    double targ {this->prob->y[n]};
                    acc += (pred==targ);
                    n_cntr += 1.0;
                }
            }
            std::cout << "Training accuracy: " << (acc/n_cntr)*100.0 << "% \n";
            acc = 0.0;
            n_cntr = 0.0;
            for (unsigned int n = 0; n<static_cast<unsigned int>(testfold_ind.size()); ++n)
            {
                double pred {this->predict(this->prob->x[testfold_ind[n]])};
                double targ {this->prob->y[testfold_ind[n]]};
                acc += (pred==targ);
                n_cntr += 1.0;
            }
            std::cout << "Test accuracy: " << (acc/n_cntr)*100.0 << "% \n";
            ++iter;
        }
        // and finally reset again
        this->reset();
        std::cout << "-------------------\n\n";
    }
    else
    {
        if(this->root == nullptr)
            std::cerr << "[warning] Model has not been constructed yet!\n";
        else
            std::cerr << "[warning] Model is in predict mode!\n";
    }
}

void HierModel::reset()
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

                // reinitialize weights and rest delta's
                visit_node->reset();
            }            
        }
    }
    else
    {
        std::cerr << "[warning] Model has not been constructed yet!\n";
    }
}

void HierModel::fit(const std::vector<unsigned int>& ign_index, const bool verbose)
{
    if (this->root != nullptr && this->prob != nullptr)
    {
        int e_cntr {0};
        while(e_cntr<this->param->ne)
        {
            double e_loss {0.0};
            double n_cntr {0.0};
            // run over each instance 
            for (unsigned int n = 0; n<static_cast<unsigned int>(this->prob->l); ++n)
            {
                if (std::find(ign_index.begin(), ign_index.end(), n) == ign_index.end())
                {
                    double i_loss {0.0};
                    feature_node* x {this->prob->x[n]};
                    std::vector<int> y {(int) this->prob->y[n]}; // our class 
                    HNode* visit_node = this->root;
                    while(!visit_node->chn.empty())
                    {
                        int ind = -1;
                        for (unsigned int i = 0; i<visit_node->chn.size(); ++i)
                        { 
                            if (std::includes(visit_node->chn[i]->y.begin(), visit_node->chn[i]->y.end(), y.begin(), y.end()) == 1)
                            {
                                ind = static_cast<int>(i);
                                break;
                            }  
                        }
                        if (ind != -1)
                        {
                            double i_p {visit_node->update(x, static_cast<long>(ind), this->param->lr)};
                            i_loss += std::log2((i_p<=EPS ? EPS : i_p));
                            visit_node = visit_node->chn[static_cast<unsigned long>(ind)];
                        }
                        else
                        {
                            std::cerr << "[error] label " << y[0] << " not found in hierarchy!\n";
                            exit(1);
                        }
                    }
                    e_loss += -i_loss;
                    n_cntr += 1;
                }
            }
            if (verbose)
                std::cout << "Epoch " << (e_cntr+1) << ": loss " << (e_loss/n_cntr) << '\n';
            ++e_cntr;
        }
        if (verbose)
            std::cout << '\n';
    }
    else
    {
        if(this->root == nullptr)
            std::cerr << "[warning] Model has not been constructed yet!\n";
        else
            std::cerr << "[warning] Model is in predict mode!\n";
    }
}

double HierModel::predict(const feature_node *x)
{
    if (root == nullptr)
    {
        std::cerr << "[warning] Model has not been constructed yet!\n";
        return -1.0;
    }
    else
    {
        HNode* visit_node = this->root;
        while(!visit_node->chn.empty())
           visit_node = visit_node->chn[visit_node->predict(x)];
        
        return static_cast<double>(visit_node->y[0]);
    }
}
    
int HierModel::getNrClass()
{
    if (root == nullptr)
    {
        std::cerr << "[warning] Model has not been constructed yet!\n";
        return 0;
    }
    else
        return this->root->y.size();
}

/*
    TODO: catch possible exceptions in this function (might become non-void eventually)
    Important: all attributes, to be saved, are required to be stored before w!
*/
void HierModel::save(const char* model_file_name)
{
    if (this->root != nullptr && this->prob != nullptr)
    {
        // create output file stream
        std::ofstream model_file;
        // TODO: add check whether below was successfull
        model_file.open(model_file_name, std::ofstream::trunc);
        // STRUCT
        model_file << "h_struct [";
        // process all except last element
        for (unsigned int i=0; i<this->prob->h_struct.size()-1; ++i)
            model_file << vecToArr(this->prob->h_struct[i]) << ',';
        // and now last element
        model_file << vecToArr(this->prob->h_struct[this->prob->h_struct.size()-1]) << "]\n";
        // #FEATURES
        model_file << "nr_feature " << this->prob->n << '\n';
        // BIAS
        model_file << "bias " << (this->prob->bias >= 0. ? 1.0 : -1.0) << '\n';
        // WEIGHTS
        model_file << "w \n";
        std::queue<HNode*> visit_list; 
        visit_list.push(this->root);
        while(!visit_list.empty())
        {
            HNode* visit_node = visit_list.front();
            // print out weights
            model_file << visit_node->getWeightVector() << '\n';
            // remove from Q
            visit_list.pop();
            // only internal nodes are going to be processed eventually (and end up in Q)
            for(auto* c : visit_node->chn)
            {
                if(!c->chn.empty())
                    visit_list.push(c);
            }            
        }
        // close file
        model_file.close();
    }
    else
    {
        if(this->root == nullptr)
            std::cerr << "[warning] Model has not been constructed yet!\n";
        else
            std::cerr << "[warning] Model is in predict mode!\n";
    }
}

void HierModel::load(const char* model_file_name)
{
    problem* prob = new problem{}; 
    //1. create prob instance, based on information in file: h_struct, nr_feature, bias
    std::ifstream in {model_file_name};
    std::string line;
    bool w_mode {0};
    // Q for storing nodes (setting weights)
    std::queue<HNode*> visit_list; 
    try
    {
        while (std::getline(in, line))
        {
            // get tokens for line (ie class and index:ftval)
            std::istringstream istr_stream {line};
            if (!w_mode)
            {
                // not yet in w mode, hence, get tokens based on ' ' 
                std::vector<std::string> tokens {std::istream_iterator<std::string> {istr_stream}, std::istream_iterator<std::string>{}};
                if (tokens[0] == "h_struct")
                    prob->h_struct = strToHierarchy(tokens[1]);
                else if(tokens[0] == "nr_feature")
                    prob->n = std::stoi(tokens[1]);
                else if(tokens[0] == "bias")
                    prob->bias = std::stod(tokens[1]);
                else
                {
                    // all required info, for tree construction, is stored in prob!
                    this->root = new HNode(*prob);
                    // and store root in Q
                    visit_list.push(this->root);
                    // set w mode
                    w_mode = 1;
                }
            }
            else
            {
                // weight line => store in current node 
                while(!visit_list.empty())
                {
                    HNode* visit_node = visit_list.front();
                    // set weights
                    visit_node->setWeightVector(line);
                    // remove from Q
                    visit_list.pop();
                    // only internal nodes are going to be processed eventually (and end up in Q)
                    for(auto* c : visit_node->chn)
                    {
                        if(!c->chn.empty())
                            visit_list.push(c);
                    }            
                }
            }
        }
    }
    catch(std::ifstream::failure e)
    {
        std::cerr << "[error] Exception " << e.what() << " catched!\n";
        exit(1);
    }
    delete prob;
}