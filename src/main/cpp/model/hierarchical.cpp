/* 
    Author: Thomas Mortier 2019-2020

    Implementation of model based on h-softmax

    TODO: implement predict_proba
    TODO: check whether backward step can be optimized (ie, no updates needed for all W's, for a given node)
*/

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

/* CONSTRUCTORS AND DESTRUCTOR */

/* will only be called on root */
HNode::HNode(const problem &prob) 
{
    // first init W matrix
    this->W = Matrix{new double*[prob.d], prob.d, 0};
    // init D vector
    this->D = Matrix{new double*[prob.d], prob.d, 0};
    // set y attribute of this node (i.e., root)
    this->y = prob.hstruct[0];
    // now construct tree
    for (unsigned long i = 1; i < prob.hstruct.size(); ++i)
        this->addChildNode(prob.hstruct[i], prob);    
}   

/* constructor for all nodes except root */
HNode::HNode(std::vector<unsigned long> y, const problem &prob) : y{y}
{
    // only init D if internal node! (TODO: optimize for internal nodes with one (internal) node as child)
    if (y.size() > 1)
    {
        // first init W matrix
        this->W = Matrix{new double*[prob.d], prob.d, 0};
        // init D vector
        this->D = Matrix{new double*[prob.d], prob.d, 0};
    }
} 

HNode::~HNode()
{
    this->free();
}

/* PUBLIC */

/*
    Predict branch, given instance.

    Arguments:
        x: feature node
    Return: 
        index branch with highest (conditional) probability
*/
unsigned long HNode::predict(const feature_node *x)
{
    unsigned long max_i {0};
    if (this->chn.size() > 1)
    {
        // forward step
        double* o {new double[this->W.k]()}; // array of exp
        // Wtx
        dgemv(1.0, const_cast<const double**>(this->W.value), x, o, this->W.k);
        // get index max
        double* max_o {std::max_element(o, o+this->W.k)}; 
        max_i = static_cast<unsigned long>(max_o-o);
        // delete
        delete[] o;
    }
    return max_i;
}

/*
    Get probability of branch, given instance.

    Arguments:
        x: feature node
        ind: index of branch
    Return: 
        probability of branch with index ind
*/
double HNode::predict(const feature_node *x, const unsigned long ind)
{
    double prob {1.0};
    if (this->chn.size() > 1)
    {
        // forward step
        double* o {new double[this->W.k]()}; // array of exp
        // Wtx
        dgemv(1.0, const_cast<const double**>(this->W.value), x, o, this->W.k);
        // apply softmax
        softmax(o, this->W.k);
        // return prob
        prob = o[ind];
        // delete
        delete[] o;
    }
    return prob;
}

/*
    Forward pass and backprop call.

    Arguments:
        x: feature node
        ind: index for branch to be updated
        lr: learning rate for SGD
        fast: whether to apply fast (but less accurate) updates or not
    Return: 
        probability for branch with index ind (needed for loss)
*/
double HNode::update(const feature_node *x, const unsigned long ind, const double lr, const bool fast)
{
    double prob {1.0};
    if (this->chn.size() > 1)
    {
        // forward step
        double* o {new double[this->W.k]()}; // array of exp
        // Wtx
        dgemv(1.0, const_cast<const double**>(this->W.value), x, o, this->W.k);
        // apply softmax
        softmax(o, this->W.k); 
        // set delta's 
        if (fast)
        {
            dvscalm((o[ind]-1), x, this->D.value, this->D.k, ind);
            // backward step
            dsubmv(lr, this->W.value, const_cast<const double**>(this->D.value), this->W.d, this->W.k, ind);
        }
        else
        {
            double t {0.0};
            for(unsigned long i=0; i<this->D.k; ++i)
            {
                if(static_cast<const unsigned long&>(i) == ind)
                    t = 1.0;
                else
                    t = 0.0;

                dvscalm((o[i]-t), x, this->D.value, this->D.k, i);
            }
            // backward step
            for (unsigned long i=0; i<this->W.k; ++i)
                dsubmv(lr, this->W.value, const_cast<const double**>(this->D.value), this->W.d, this->W.k, i);
        }    
        prob = o[ind];
        // delete
        delete[] o;
    }
    return prob;
}

/* reinitialize W */
void HNode::reset()
{
    // reinitialize W
    if (this->chn.size() > 1)
        initUW(static_cast<double>(-1.0/this->W.d), static_cast<double>(1.0/this->W.d), this->W.value, this->W.d, this->W.k);
}

/*
    Add new node with label set y to current node.

    Arguments:
        y: label set for new node to be added
        prob: problem instance
*/
void HNode::addChildNode(std::vector<unsigned long> y, const problem &prob)
{
    // todo: optimize?
    // check if leaf or internal node 
    if (this->chn.size() > 0)
    {
        // check if y is a subset of one of the children
        long ind = -1;
        for (unsigned long i = 0; i < this->chn.size(); ++i)
        {
            if (std::includes(this->chn[i]->y.begin(), this->chn[i]->y.end(), y.begin(), y.end()) == 1)
            {
                ind = static_cast<long>(i);
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
                for (unsigned long i=0; i<prob.d; ++i)
                {
                    this->W.value[i] = new double[this->chn.size()]{0};
                    this->D.value[i] = new double[this->chn.size()]{0};
                }
            }
            // set k size attribute
            this->W.k = this->chn.size();
            this->D.k = this->chn.size();
        }
    }
    else
    { 
        // no children yet, hence, put in children list
        HNode* new_node = new HNode{y, prob};
        this->chn.push_back(new_node);
    }
}

/* deallocate memory (W and D) for current node */
void HNode::free()
{
    if (this->chn.size() > 1)
    { 
        for (unsigned long i = 0; i < this->W.d; ++i)
        {
            delete[] this->W.value[i];
            delete[] this->D.value[i];
        }
        delete[] this->W.value;
        delete[] this->D.value;
    }
}  

/* returns weights for current node in string representation (row-major order, space separated) */
std::string HNode::getWeightVector()
{
    std::string ret_arr {""};
    if (this->chn.size() > 1)
    {
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
    }
    return ret_arr;
}

/* set weights in string representation (row-major order, space separated) */ 
void HNode::setWeightVector(std::string w_str)
{
    if (this->chn.size() > 1)
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
}

/* print node */
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

/* CONSTRUCTOR AND DESTRUCTOR */

/* constructor (training mode) */
HierModel::HierModel(const problem* prob) : Model(prob)
{
    // construct tree 
    root = new HNode(*prob);
}

/* constructor (predict mode) */
HierModel::HierModel(const char* model_file_name) : Model(model_file_name)
{
    std::cout << "Loading model from " << model_file_name << "...\n";
    this->load(model_file_name);
}

HierModel::~HierModel()
{
    std::queue<HNode*> visit_list; 
    visit_list.push(this->root);
    while (!visit_list.empty())
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

/* PUBLIC */

/* print structure of hierarchy */
void HierModel::printStruct()
{
    this->root->print();
}

/* print some general information about model */
void HierModel::printInfo(const bool verbose)
{
    std::cout << "---------------------------------------------------\n";
    std::cout << "[info] Hierarchical model: \n";
    std::cout << "---------------------------------------------------\n";
    std::cout << "  * Number of features              = " << this->getNrFeatures() << '\n';
    std::cout << "  * Number of classes               = " << this->getNrClass() << '\n';
    if (verbose)
    {
        std::cout << "  * Structure =\n";
        this->printStruct();
    }
    std::cout << "---------------------------------------------------\n\n";
}

/* k-fold cross-validation */
void HierModel::performCrossValidation(unsigned int k)
{
    if (this->prob != nullptr)
    {
        std::cout << "---- " << k << "-Fold CV ----\n";
        // first create index vector
        std::vector<unsigned long> ind_arr;
        for(unsigned long i=0; i<this->prob->n; ++i)
            ind_arr.push_back(i);
        // now shuffle index vector 
        auto rng = std::default_random_engine {};
        std::shuffle(std::begin(ind_arr), std::end(ind_arr), rng);
        // calculate size of each test fold
        unsigned long ns_fold {this->prob->n/static_cast<unsigned long>(k)};
        // start kfcv
        unsigned int iter {0};
        while (iter < k)
        {
            std::cout << "FOLD " << iter+1 << '\n';
            // first clear weights 
            this->reset();
            // extract test fold indices
            std::vector<unsigned long>::const_iterator i_start = ind_arr.begin() + static_cast<long>(static_cast<unsigned long>(iter)*ns_fold);
            std::vector<unsigned long>::const_iterator i_stop = ind_arr.begin() + static_cast<long>(static_cast<unsigned long>((iter+1))*ns_fold);
            std::vector<unsigned long> testfold_ind(i_start, i_stop);
            // now start fitting 
            this->fit(testfold_ind, 0);
            // and validate on training and test fold
            double acc {0.0};
            double n_cntr {0.0};
            for (unsigned long n = 0; n<this->prob->n; ++n)
            {
                if (std::find(testfold_ind.begin(), testfold_ind.end(), n) == testfold_ind.end())
                {
                    unsigned long pred {this->predict(this->prob->X[n])};
                    unsigned long targ {this->prob->y[n]};
                    acc += (pred==targ);
                    n_cntr += 1.0;
                }
            }
            std::cout << "Training accuracy: " << (acc/n_cntr)*100.0 << "% \n";
            acc = 0.0;
            n_cntr = 0.0;
            for (unsigned long n = 0; n<testfold_ind.size(); ++n)
            {
                unsigned long pred {this->predict(this->prob->X[testfold_ind[n]])};
                unsigned long targ {this->prob->y[testfold_ind[n]]};
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
        std::cerr << "[warning] Model is in predict mode!\n";
    }
}

/* reset all nodes model */
void HierModel::reset()
{
    if (root != nullptr)
    {
        std::queue<HNode*> visit_list; 
        visit_list.push(this->root);
        while (!visit_list.empty())
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

/* fit on data (in problem instance), while ignoring instances with ind in ign_index */
void HierModel::fit(const std::vector<unsigned long>& ign_index, const bool verbose)
{
    std::cout << "Fit model...\n";
    if (this->root != nullptr && this->prob != nullptr)
    {
        unsigned int e_cntr {0};
        while (e_cntr < this->prob->ne)
        {
            double e_loss {0.0};
            double n_cntr {0.0};
            // run over each instance 
            for (unsigned long n = 0; n<this->prob->n; ++n)
            {
                //std::cout << "Fit on instance " << n << " during epoch " << e_cntr << " ...\n";
                if (std::find(ign_index.begin(), ign_index.end(), n) == ign_index.end())
                {
                    double i_loss {0.0};
                    feature_node* x {this->prob->X[n]};
                    std::vector<unsigned long> y {this->prob->y[n]}; // our class 
                    HNode* visit_node = this->root;
                    while (!visit_node->chn.empty())
                    {
                        long ind = -1;
                        for (unsigned long i = 0; i<visit_node->chn.size(); ++i)
                        { 
                            if (std::includes(visit_node->chn[i]->y.begin(), visit_node->chn[i]->y.end(), y.begin(), y.end()) == 1)
                            {
                                ind = static_cast<long>(i);
                                break;
                            }  
                        }
                        if (ind != -1)
                        {
                            double i_p {visit_node->update(x, static_cast<unsigned long>(ind), this->prob->lr, this->prob->fast)};
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

/*
    Return label (class) of leaf node with highest probability mass.

    Arguments:
        x: feature node
    Return: 
        label (class) of leaf node 
*/
unsigned long HierModel::predict(const feature_node *x)
{
    if (root == nullptr)
    {
        std::cerr << "[warning] Model has not been constructed yet!\n";
        return 0;
    }
    else
    {
        HNode* visit_node = this->root;
        while (!visit_node->chn.empty())
           visit_node = visit_node->chn[visit_node->predict(x)];
        
        return visit_node->y[0];
    }
}

/*
    Calculate probability masses for one or more leaf nodes.

    Arguments:
        x: feature node
        y: vector of labels of leaf nodes for which to calculate probability mass
        p: vector of probabilities
    Return: 
        vector of probabilities
*/
std::vector<double> HierModel::predict_proba(const feature_node* x, const std::vector<unsigned long> yv)
{
    std::vector<double> probs; 
    // run over all labels for which we need to calculate probs
    for (unsigned long ye : yv)
    {
        // transform y to singleton set/vector
        std::vector<unsigned long> y {ye};
        // begin at root
        HNode* visit_node = this->root;
        double prob {1.0};
        while (!visit_node->chn.empty())
        {
            long ind = -1;
            for (unsigned long i = 0; i<visit_node->chn.size(); ++i)
            { 
                if (std::includes(visit_node->chn[i]->y.begin(), visit_node->chn[i]->y.end(), y.begin(), y.end()) == 1)
                {
                    ind = static_cast<long>(i);
                    break;
                }  
            }
            if (ind != -1)
            {
                // (i_p<=EPS ? EPS : i_p) in case of underflow! (TODO: remove)
                prob *= visit_node->predict(x, static_cast<unsigned long>(ind));
                visit_node = visit_node->chn[static_cast<unsigned long>(ind)];
            }
            else
            {
                std::cerr << "[error] label " << ye << " not found!\n";
                exit(1);
            }
        }
        probs.push_back(prob);
    }
    return probs;
}

/* get number of classes */
unsigned long HierModel::getNrClass()
{
    if (this->prob != nullptr)
        return this->prob->hstruct[0].size();
    else
        return this->root->y.size();
}

/* get number of features (bias included) */ 
unsigned long HierModel::getNrFeatures()
{
    if (this->prob != nullptr)
        return this->prob->d;
    else
        return this->root->W.d; 
}

/* save model to file */
void HierModel::save(const char* model_file_name)
{
    std::cout << "Saving model to " << model_file_name << "...\n";
    if (this->prob != nullptr)
    {
        // create output file stream
        std::ofstream model_file;
        model_file.open(model_file_name, std::ofstream::trunc);
        // STRUCT
        model_file << "struct [";
        // process all except last element
        for (unsigned int i=0; i<this->prob->hstruct.size()-1; ++i)
            model_file << vecToArr(this->prob->hstruct[i]) << ',';
        // and now last element
        model_file << vecToArr(this->prob->hstruct[this->prob->hstruct.size()-1]) << "]\n";
        // #features
        model_file << "nr_feature " << this->prob->d << '\n';
        // bias
        model_file << "bias " << (this->prob->bias >= 0. ? 1.0 : -1.0) << '\n';
        // weights
        model_file << "w \n";
        std::queue<HNode*> visit_list; 
        visit_list.push(this->root);
        while (!visit_list.empty())
        {
            HNode* visit_node = visit_list.front();
            // store weights
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
        std::cerr << "[warning] Model is in predict mode!\n";
}

/* load model from file */
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
                if (tokens[0] == "struct")
                    prob->hstruct = strToHierarchy(tokens[1]);
                else if(tokens[0] == "nr_feature")
                    prob->d = static_cast<unsigned long>(std::stol(tokens[1]));
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
    catch(std::ifstream::failure e)
    {
        std::cerr << "[error] Exception " << e.what() << " catched!\n";
        exit(1);
    }
    delete prob;
}