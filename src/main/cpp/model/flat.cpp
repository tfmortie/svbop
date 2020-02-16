/* 
    Author: Thomas Mortier 2019-2020

    Implementation of model with softmax

    TODO: comments
    TODO: optimize (allow sparse features (feature_node))!
    TODO: implement predict_proba
*/

#include "model/model.h"
#include "model/flat.h"
#include "data.h"
#include "utils.h"
#include <sstream>
#include <iomanip>
#include <random>
#include <fstream>
#include <queue>


/* CONSTRUCTOR AND DESTRUCTOR */

/* constructor (training mode) */
FlatModel::FlatModel(const problem* prob) : Model(prob)
{
    // create W & D matrix
    this->W = Matrix{new double*[prob->d], prob->d, prob->hstruct[0].size()};
    this->D = Matrix{new double*[prob->d], prob->d, prob->hstruct[0].size()};
    for (unsigned long i=0; i<prob->d; ++i)
    {
        this->W.value[i] = new double[prob->hstruct[0].size()];
        this->D.value[i] = new double[prob->hstruct[0].size()]{0};
    }
    // initialize W matrix
    initUW(static_cast<double>(-1.0/this->W.d), static_cast<double>(1.0/this->W.d), this->W.value, this->W.d, this->W.k);
}

/* constructor (predict mode) */
FlatModel::FlatModel(const char* model_file_name) : Model(model_file_name)
{
    this->load(model_file_name);
}

FlatModel::~FlatModel()
{
    this->free();
}

/* PRIVATE */

/*
    Forward pass and backprop call.

    Arguments:
        x: feature node
        y: class for which to calculate the probability (needed for loss)
        lr: learning rate for SGD
    Return: 
        probability for class y (needed for loss)
*/
double FlatModel::update(const feature_node* x, const unsigned long y, const double lr)
{
    // forward step
    double* o {new double[this->W.k]()}; // array of exp
    // Wtx
    dgemv(1.0, const_cast<const double**>(this->W.value), x, o, this->W.k);
    // apply softmax
    softmax(o, this->W.k); 
    // set delta's 
    double t {0.0};
    for(unsigned long i=0; i<this->D.k; ++i)
    {
        if (i+1 == y)
            t = 1.0;
        else
            t = 0.0;

        dvscalm((o[i]-t), x, this->D.value, this->D.k, i);
    }
    // backward step
    this->backward(x, lr);
    double p {o[y-1]};
    // delete
    delete[] o;
    return p;
}

/*
    Backprop.

    Arguments:
        x: feature node
        lr: learning rate for SGD
*/
void FlatModel::backward(const feature_node* x, const double lr)
{
    for (unsigned long i=0; i<this->W.k; ++i)
        dsubmv(lr, this->W.value, const_cast<const double**>(this->D.value), this->W.d, this->W.k, i);
}

/* returns weights in string representation (row-major order, space separated) */
std::string FlatModel::getWeightVector()
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

/* set weights in string representation (row-major order, space separated) */ 
void FlatModel::setWeightVector(std::string w_str)
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

/* deallocate memory (W and D) for current node */
void FlatModel::free()
{
    for (unsigned long i = 0; i < this->W.d; ++i)
    {
        delete[] this->W.value[i];
        delete[] this->D.value[i];
    }
    delete[] this->W.value;
    delete[] this->D.value;
}

/* PUBLIC */ 

/* print structure of classification problem */
void FlatModel::printStruct()
{
    if (this->prob == nullptr)
    {
        std::cout << "[";
        for (unsigned long i=0; i<this->W.k-1; ++i)
            std::cout << i << ',';
        std::cout << this->W.k << "]\n";
    }
    else
        std::cout << vecToArr(this->prob->hstruct[0]) << '\n';
}

/* print some general information about model */
void FlatModel::printInfo(const bool verbose)
{
    std::cout << "---------------------------------------------------\n";
    std::cout << "[info] Flat model: \n";
    std::cout << "---------------------------------------------------\n";
    std::cout << "  * Number of features              = " << this->W.d << '\n';
    std::cout << "  * Number of classes               = " << this->W.k << '\n';   
    if (verbose)
    {
        std::cout << "  * Structure =\n";
        this->printStruct();
    }
    std::cout << "---------------------------------------------------\n\n";
}

/* k-fold cross-validation */
void FlatModel::performCrossValidation(unsigned int k)
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
        std::cerr << "[warning] Model is in predict mode!\n";
}

/* reset model (ie, weights) */
void FlatModel::reset()
{
    // reinitialize W
    initUW(static_cast<double>(-1.0/this->W.d), static_cast<double>(1.0/this->W.d), this->W.value, this->W.d, this->W.k);
}

/* fit on data (in problem instance), while ignoring instances with ind in ign_index */
void FlatModel::fit(const std::vector<unsigned long>& ign_index, const bool verbose)
{
    std::cout << "Fit model...\n";
    if (this->prob != nullptr)
    {
        unsigned int e_cntr {0};
        while (e_cntr < this->prob->ne)
        {
            double e_loss {0.0};
            double n_cntr {0.0};
            // run over each instance 
            for (unsigned long n = 0; n<this->prob->n; ++n)
            {
                std::cout << "Fit on instance " << n << " during epoch " << e_cntr << " ...\n";
                if (std::find(ign_index.begin(), ign_index.end(), n) == ign_index.end())
                {
                    feature_node* x {this->prob->X[n]};
                    unsigned long y {this->prob->y[n]}; // our class 
                    double i_p {this->update(x, y, this->prob->lr)};
                    double i_loss {std::log2((i_p<=EPS ? EPS : i_p))};
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
        std::cerr << "[warning] Model is in predict mode!\n";
}

/*
    Return class with highest probability mass.

    Arguments:
        x: feature node
    Return: 
        class
*/
unsigned long FlatModel::predict(const feature_node* x)
{
    // forward step
    double* o {new double[this->W.k]()}; // array of exp
    // Wtx
    //dgemv(1.0, const_cast<const double**>(this->W.value), x_arr, o, this->W.d, this->W.k);
    dgemv(1.0, const_cast<const double**>(this->W.value), x, o, this->W.k);
    // get index max
    double* max_o {std::max_element(o, o+this->W.k)}; 
    unsigned long max_i {static_cast<unsigned long>(max_o-o)};
    // delete
    delete[] o;
    return max_i+1;
}

/*
    Calculate probability masses for one or more classes.

    Arguments:
        x: feature node
        yv: vector of classes for which to calculate probability mass
        p: vector of probabilities
    Return: 
        vector of probabilities
*/
std::vector<double> FlatModel::predict_proba(const feature_node* x, const std::vector<unsigned long> yv)
{
    std::vector<double> probs; 
    // run over all labels for which we need to calculate probs
    for (unsigned long y : yv)
    {
        double prob {0.0};
        // forward step
        double* o {new double[this->W.k]()}; // array of exp
        // Wtx
        dgemv(1.0, const_cast<const double**>(this->W.value), x, o, this->W.k);
        // apply softmax
        softmax(o, this->W.k);
        // get prob
        prob = o[y-1];
        // delete
        delete[] o;
        probs.push_back(prob);
    }
    return probs;
}

/* get number of classes */
unsigned long FlatModel::getNrClass()
{
    return this->W.k;
}

/* get number of features (bias included) */ 
unsigned long FlatModel::getNrFeatures()
{
    return this->W.d;
}

/* save model to file */
void FlatModel::save(const char* model_file_name)
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
        model_file << "bias " << (this->prob->bias >= 0.0 ? 1.0 : -1.0) << '\n';
        // weights
        model_file << "w \n";
        model_file << this->getWeightVector() << '\n';       
        // close file
        model_file.close();
    }
    else
        std::cerr << "[warning] Model is in predict mode!\n";
}

/* load model from file */
void FlatModel::load(const char* model_file_name)
{
    std::cout << "Loading model from " << model_file_name << "...\n";
    problem* prob = new problem{}; 
    //1. create prob instance, based on information in file: h_struct, nr_feature, bias
    std::ifstream in {model_file_name};
    std::string line;
    bool w_mode {0};
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
                    prob->d = static_cast<unsigned long>(std::stoi(tokens[1]));
                else if(tokens[0] == "bias")
                    prob->bias = std::stod(tokens[1]);
                else
                {
                    // set w mode
                    w_mode = 1;
                }
            }
            else
            {
                // first create W & D matrix
                this->W = Matrix{new double*[prob->d], prob->d, prob->hstruct[0].size()};
                this->D = Matrix{new double*[prob->d], prob->d, prob->hstruct[0].size()};
                for (unsigned long i=0; i<prob->d; ++i)
                {
                    this->W.value[i] = new double[prob->hstruct[0].size()]{0};
                    this->D.value[i] = new double[prob->hstruct[0].size()]{0};
                }
                this->setWeightVector(line);         
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