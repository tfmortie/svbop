/* 
    Author: Thomas Mortier 2019-2020

    Header shared between different models
*/

#ifndef MODEL_U
#define MODEL_U

#include <iostream>
#include <vector>
#include "Eigen/Dense"
#include "Eigen/SparseCore"

/* type of model */
enum class ModelType {
    SOFTMAX,
    HSOFTMAXS,
    HSOFTMAXF
};

/* struct which contains learning problem specific information */
struct problem
{
    /* DATA */
	unsigned long n, d; /* number of instances and features (including bias) */
    std::vector<std::vector<unsigned long>> hstruct; /* structure classification problem */
    std::vector<unsigned long> y; /* vector with classes */
    std::vector<Eigen::SparseVector<double>> X; /* vector with sparse vectors */
	double bias; /* < 0 if no bias */
    /* LEARNING */
    unsigned int ne; /* number of epochs for training (SGD) */
    double lr; /* learning rate for training (SGD) */
    bool fast; /* fast backprop (h-softmax) */
};

/* superclass model */
class Model 
{
    protected:
        const problem* prob;
        
    public:
        Model(const problem* prob) : prob{prob} {};
        Model(const char* model_file_name) : prob{nullptr} {};
        virtual ~Model() {};

        virtual void printStruct() = 0;
        virtual void printInfo(const bool verbose = 0) = 0;
        virtual void performCrossValidation(unsigned int k) = 0;
        virtual void reset() = 0;
        virtual void fit(const std::vector<unsigned long>& ign_index = {}, const bool verbose = 1) = 0;
        virtual unsigned long predict(const Eigen::SparseVector<double>& x) = 0;
        virtual std::vector<double> predict_proba(const Eigen::SparseVector<double>& x, const std::vector<unsigned long> ind = {}) = 0;
        virtual unsigned long getNrClass() = 0;
        virtual unsigned long getNrFeatures() = 0;
        virtual void save(const char* model_file_name) = 0;
        virtual void load(const char* model_file_name) = 0; 
};

#endif