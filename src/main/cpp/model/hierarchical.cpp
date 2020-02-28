/* 
    Author: Thomas Mortier 2019-2020

    Implementation of model based on h-softmax
*/

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
#include <functional>
#include "model/hierarchical.h"
#include "data.h"
#include "model/mmath.h"
#include "Eigen/Dense"
#include "Eigen/SparseCore"

/* CONSTRUCTORS AND DESTRUCTOR */

/* will only be called on root */
HNode::HNode(const problem &prob) 
{
    // first init W matrix
    this->W = Eigen::MatrixXd::Random(prob.d, 1);
    // init D, M and V matrices (M and V for Adam)
    this->D = Eigen::MatrixXd::Zero(prob.d, 1);
    this->M = Eigen::MatrixXd::Zero(prob.d, 1);
    this->V = Eigen::MatrixXd::Zero(prob.d, 1);
    // set y attribute of this node (i.e., root)
    this->y = prob.hstruct[0];
    // now construct tree
    for (unsigned long i = 1; i < prob.hstruct.size(); ++i)
        this->addChildNode(prob.hstruct[i], prob);    
}   

/* constructor for all nodes except root */
HNode::HNode(std::vector<unsigned long> y, const problem &prob) : y{y}
{
    // only init D if internal node!
    if (y.size() > 1)
    {
        // first init W matrix
        this->W = Eigen::MatrixXd::Random(prob.d, 1);
        // init D, M and V matrices (M and V for Adam)
        this->D = Eigen::MatrixXd::Zero(prob.d, 1);
        this->M = Eigen::MatrixXd::Zero(prob.d, 1);
        this->V = Eigen::MatrixXd::Zero(prob.d, 1);
    }
} 

/* PUBLIC */

/*
    Predict branch, given instance.

    Arguments:
        x: sparse feature vector
    Return: 
        index branch with highest (conditional) probability
*/
unsigned long HNode::predict(const Eigen::SparseVector<double>& x)
{
    unsigned long max_i {0};
    if (this->chn.size() > 1)
    {
        // forward step (Wtx)
        Eigen::VectorXd o = this->W.transpose() * x;
        // apply softmax
        softmax(o);
        Eigen::VectorXd::Index max_ind;
        o.maxCoeff(&max_ind);
        max_i = max_ind;
    }
    return max_i;
}

/*
    Get probability of branch, given instance.

    Arguments:
        x: sparse feature vector
        ind: index of branch
    Return: 
        probability of branch with index ind
*/
double HNode::predict(const Eigen::SparseVector<double>& x, const unsigned long ind)
{
    double prob {1.0};
    if (this->chn.size() > 1)
    {
        // forward step (Wtx)
        Eigen::VectorXd o = this->W.transpose() * x;
        // apply softmax
        softmax(o);
        prob = o[ind];
    }
    return prob;
}

/*
    Forward pass and backprop call.

    Arguments:
        x: sparse feature vector
        ind: index for branch to be updated
        t: time step in case of Adam optimization
    Return: 
        probability for branch with index ind (needed for loss)
*/
double HNode::update(const Eigen::SparseVector<double>& x, const unsigned long ind, const problem& prob, const unsigned long t)
{
    double p {1.0};
    if (this->chn.size() > 1)
    {
        // forward step (Wtx)
        Eigen::VectorXd o = this->W.transpose() * x;
        // apply softmax
        softmax(o);
        // calculate derivatives and backprop 
        if (prob.fast)
        {
            // set derivatives
            this->D.col(ind) += (o[ind]-1.0)*x;
            // and update in case we have processed a mini-batch
            if (t % prob.batchsize)
            {
                // and update in case we have processed a mini-batch
                // calculate the average gradients
                this->D.col(ind) = this->D.col(ind)/prob.batchsize;
                if (prob.optim == OptimType::SGD)
                    sgd(this->W, this->D, prob.lr);
                else
                    adam(this->W, this->D, this->M, this->V, prob.lr, t, ind);
                this->D.setZero();
            }
        }
        else
        {
            // set derivatives
            dvscalm(D, o, ind, x);
            // and update in case we have processed a mini-batch
            if (t % prob.batchsize == 0)
            {
                // and update
                if (prob.optim == OptimType::SGD)
                    sgd(this->W, this->D, prob.lr);
                else
                    adam(this->W, this->D, this->M, this->V, prob.lr, t, -1);
                this->D.setZero();
            }
        }    
        p = o[ind];
    }
    return p;
}

/* reinitialize W */
void HNode::reset()
{
    // reinitialize W
    if (this->chn.size() > 1)
        inituw(this->W, static_cast<double>(-1.0/this->W.rows()), static_cast<double>(1.0/this->W.rows()));
        // and D,M and V (M,V for Adam)
        this->D.setZero();
        this->M.setZero();
        this->V.setZero();
}

/*
    Add new node with label set y to current node.

    Arguments:
        y: label set for new node to be added
        prob: problem instance
*/
void HNode::addChildNode(std::vector<unsigned long> y, const problem &prob)
{
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
                // allocate W, D, M and V matrix (M and V for Adam)
                this->W.resize(this->W.rows(), this->chn.size());
                this->D.resize(this->D.rows(), this->chn.size());
                this->M.resize(this->M.rows(), this->chn.size());
                this->V.resize(this->V.rows(), this->chn.size());
                // initialize W matrix
                inituw(this->W, static_cast<double>(-1.0/this->W.rows()), static_cast<double>(1.0/this->W.rows()));
            }
        }
    }
    else
    { 
        // no children yet, hence, put in children list
        HNode* new_node = new HNode{y, prob};
        this->chn.push_back(new_node);
    }
}

/* returns weights for current node in string representation (row-major order, space separated) */
std::string HNode::getWeightVector()
{
    std::string ret_arr {""};
    if (this->chn.size() > 1)
    {
        // process all elements in row-major order
        for (unsigned long i=0; i<this->W.rows(); ++i)
        {
            for (unsigned long j=0; j<this->W.cols(); ++j)
            {
                std::stringstream str_stream;
                str_stream << std::to_string(this->W(i,j));
                ret_arr += str_stream.str();
                if ((i!=this->W.rows()-1) || (j!=this->W.cols()-1))
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
        for (unsigned long i=0; i<this->W.rows(); ++i)
        {
            for (unsigned long j=0; j<this->W.cols(); ++j)
                this->W(i,j) = std::stod(tokens[(i*this->W.cols())+j]);
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
HierModel::HierModel(problem* prob) : Model(prob)
{
    // construct tree 
    root = new HNode(*prob);
}

/* constructor (predict mode) */
HierModel::HierModel(const char* model_file_name, problem* prob) : Model(model_file_name, prob)
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

/* fit on data (in problem instance), while validating on instances with ind in ign_index (if applicable) */
void HierModel::fit(const std::vector<unsigned long>& ign_index, const bool verbose)
{
    std::cout << "---------------------------------------------------------------------------------\n";
    if (ign_index.size() != 0)
        std::cout << "Fit model on train/val (" << this->prob->n-ign_index.size() << '/' << ign_index.size() << ") data ...\n";
    else
        std::cout << "Fit model on train (" << this->prob->n << ") data ...\n";
    if (this->root != nullptr)
    {
        std::cout << "---------------------------------------------------------------------------------\n";
        std::cout << "* #Features: " << this->prob->d << '\n';
        std::cout << "* #Classes: " << this->prob->hstruct[0].size() << '\n';
        std::cout << "* #Epochs: " << this->prob->ne << '\n';
        std::cout << "* Mini-batch size: " << this->prob->batchsize << '\n';
        std::cout << "* Patience: " << this->prob->patience << '\n';
        if (this->prob->optim == OptimType::SGD)
            std::cout << "* Optimizer: SGD\n";
        else
            std::cout << "* Optimizer: Adam\n";
        std::cout << "* Learning rate: " << this->prob->lr << '\n';
        std::cout << "---------------------------------------------------------------------------------\n";
        unsigned int e_cntr {0};
        int patience_counter {0};
        double prev_best_loss {std::numeric_limits<double>::max()};
        unsigned long t {0}; // in case of Adam optimization
        // reset W,D,M,V
        this->reset();
        while (e_cntr < this->prob->ne)
        {
            // init. train/holdout loss and counter
            double e_loss_train {0.0};
            double e_loss_holdout {0.0};
            double n_cntr_train {0.0};
            double n_cntr_holdout {0.0};
            // run over each instance 
            for (unsigned long n = 0; n<this->prob->n; ++n)
            {
                if (std::find(ign_index.begin(), ign_index.end(), n) == ign_index.end())
                {
                    t += 1;
                    double i_loss {0.0};
                    Eigen::SparseVector<double> x {this->prob->X[n]};
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
                            double i_p {visit_node->update(x, static_cast<unsigned long>(ind), *this->prob, t)};
                            i_loss += std::log2((i_p<=EPS ? EPS : i_p));
                            visit_node = visit_node->chn[static_cast<unsigned long>(ind)];
                        }
                        else
                        {
                            std::cerr << "[error] label " << y[0] << " not found in hierarchy!\n";
                            exit(1);
                        }
                    }
                    e_loss_train += -i_loss;
                    n_cntr_train += 1;
                }
                else
                {
                    // holdout example, hence, only compute loss
                    Eigen::SparseVector<double> x {this->prob->X[n]};
                    std::vector<unsigned long> yv{this->prob->y[n]}; // our class
                    double i_p{this->predict_proba(x,yv)[0]};
                    double i_loss {std::log2(i_p<=EPS ? EPS : i_p)};
                    e_loss_holdout += -i_loss;
                    n_cntr_holdout += 1;
                }
            }
            // average training loss
            e_loss_train = e_loss_train/n_cntr_train;
            if (ign_index.size() != 0)
            {
                // also calculate average holdout loss
                e_loss_holdout = e_loss_holdout/n_cntr_holdout;
                // check if we have an improvement, compared to previous epoch loss
                if (e_loss_holdout < prev_best_loss)
                {
                    prev_best_loss = e_loss_holdout;
                    patience_counter = 0; // reset patience counter
                }
                else
                {
                    // increase patience counter
                    patience_counter += 1;
                }
                // check if we can early stop
                if (patience_counter == this->prob->patience)
                {
                    std::cout << "Eearly stopping at epoch " << (e_cntr+1) << ": train loss " << e_loss_train << "  -  val loss  " << e_loss_holdout << '\n';
                    break;
                }
                else
                {
                    if (verbose)
                        std::cout << "Epoch " << (e_cntr+1) << ": train loss "<< e_loss_train << "  -  val loss  " << e_loss_holdout << '\n';
                }
            }
            else
            {
                if (verbose)
                    std::cout << "Epoch " << (e_cntr+1) << ": loss "<< e_loss_train << '\n';
            }
            ++e_cntr;
        }
        if (verbose)
            std::cout << '\n';
    }
    else
    {
        std::cerr << "[warning] Model has not been constructed yet!\n";
    }
}

/*
    Return label (class) of leaf node with highest probability mass.

    Arguments:
        x: feature node
    Return: 
        label (class) of leaf node 
*/
unsigned long HierModel::predict(const Eigen::SparseVector<double>& x)
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
std::vector<double> HierModel::predict_proba(const Eigen::SparseVector<double>& x, const std::vector<unsigned long> yv)
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

/* 
    Implementation for unrestricted bayes-optimal predictor.
    See https://arxiv.org/abs/1906.08129. 
*/
std::vector<unsigned long> HierModel::predict_ubop(const Eigen::SparseVector<double>& x)
{
    // initalize prediction set, with probability and expected utility
    std::vector<unsigned long> set;
    double set_prob {0.0};
    double set_eu {0.0};
    // initialize priority queue that sorts in decreasing order of probability mass of nodes
    std::priority_queue<QueueNode> q;
    // add root 
    q.push({this->root, 1.0});
    while (!q.empty())
    {
        // get current (prob, HNode*)
        QueueNode current {q.top()};
        q.pop();
        // check if current node is a leaf 
        if (current.node->chn.size() == 0)
        {
            // push class of leaf to prediction set and add probability
            set.push_back(current.node->y[0]);
            set_prob += current.prob;
            // compute utility according to Eq. (5)
            double current_eu {set_prob*g(set, this->prob->utility)};
            // check if current solution is worse than best solution so far (early stopping criterion)
            if (current_eu < set_eu)
            {
                // remove last element from set (because previous one was optimal) and break
                set.pop_back();
                break;
            }
            else
            {
                // set new optimal expected utility and pop first element from priority queue
                set_eu = current_eu;
            }
        }
        else
        {
            // we are at an internal node: add children to priority queue
            // check if we are dealing with an internal node and single child
            if (current.node->y.size() == 1)
                q.push({current.node->chn[0], current.prob*1.0});
            else
            {
                // forward step (Wtx)
                Eigen::VectorXd o = current.node->W.transpose() * x;
                // apply softmax
                softmax(o);
                for (unsigned long i = 0; i<current.node->chn.size(); ++i)
                {
                    // (recursively) calculate probability mass of child node
                    HNode* c_node {current.node->chn[i]};
                    double c_node_prob {current.prob*o(i)};
                    // and add to priority queue
                    q.push({c_node, c_node_prob});
                }
            }
        }
    }
    return set;
}

/* 
    Implementation for restricted bayes-optimal predictor.
    See https://arxiv.org/abs/1906.08129. 
*/
std::vector<unsigned long> HierModel::predict_rbop(const Eigen::SparseVector<double>& x)
{
    // initalize optimal prediction set and corresponding expected utility 
    std::vector<unsigned long> opt_set;
    double opt_set_eu {0.0};
    // initialize priority queue that sorts in decreasing order of probability mass of nodes
    std::priority_queue<QueueNode> q;
    // add root 
    q.push({this->root, 1.0});
    while (!q.empty())
    {
        // get current (prob, HNode*)
        QueueNode current = q.top();   
        q.pop();
        // calculate expected utility for our current node
        double cur_set_eu {current.prob*g(current.node->y, this->prob->utility)};
        // set new optimal solution, in case we have an improvement
        if (cur_set_eu > opt_set_eu)
        {
            opt_set = current.node->y;
            opt_set_eu = cur_set_eu;
        }
        // check if we are at a leaf node (early stopping criterion)
        if (current.node->chn.size() == 0)
        {
            break;
        }
        else
        {
            // we are at an internal node: add children to priority queue
            // check if we are dealing with an internal node and single child
            if (current.node->y.size() == 1)
                q.push({current.node->chn[0], current.prob*1.0});
            else
            {
                // forward step (Wtx)
                Eigen::VectorXd o = current.node->W.transpose() * x;
                // apply softmax
                softmax(o);
                for (unsigned long i = 0; i<current.node->chn.size(); ++i)
                {
                    // (recursively) calculate probability mass of child node
                    HNode* c_node {current.node->chn[i]};
                    double c_node_prob {current.prob*o(i)};
                    // and add to priority queue
                    q.push({c_node, c_node_prob});
                }
            }
        }
    }
    return opt_set;
}

/* get number of classes */
unsigned long HierModel::getNrClass()
{
    return this->prob->hstruct[0].size();
}

/* get number of features (bias included) */ 
unsigned long HierModel::getNrFeatures()
{
    return this->prob->d;
}

/* save model to file */
void HierModel::save(const char* model_file_name)
{
    std::cout << "Saving model to " << model_file_name << "...\n";
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

/* load model from file */
void HierModel::load(const char* model_file_name)
{
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
                    this->prob->hstruct = strToHierarchy(tokens[1]);
                else if(tokens[0] == "nr_feature")
                    this->prob->d = static_cast<unsigned long>(std::stol(tokens[1]));
                else if(tokens[0] == "bias")
                    this->prob->bias = std::stod(tokens[1]);
                else
                {
                    // all required info, for tree construction, is stored in prob!
                    this->root = new HNode(*this->prob);
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
}