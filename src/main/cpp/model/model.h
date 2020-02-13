/* 
    Author: Thomas Mortier 2019-2020

    Header shared between different models
*/

#ifndef MODEL_U
#define MODEL_U

#include <vector>

/* type of model */
enum class ModelType {
    SOFTMAX,
    HSOFTMAX
};

/* struct which allows for sparse feature representations */
struct feature_node
{
	long index;
	double value;
};

/* struct which contains learning problem specific information */
struct problem
{
    /* DATA */
	unsigned long n, d; /* number of instances and features (including bias) */
    std::vector<std::vector<unsigned long>> hstruct; /* structure classification problem */
	unsigned long *y; /* vector with classes */
	feature_node **X; /* array with instances */
	double bias; /* < 0 if no bias */
    /* LEARNING */
    unsigned int ne; /* number of epochs for training (SGD) */
    double lr; /* learning rate for training (SGD) */
};

/* matrix container for weight and delta matrices */
struct Matrix
{
    double** value; /* should be D x K */
    unsigned long d; /* D */
    unsigned long k; /* K */
};

/* superclass model */
class Model 
{
    protected:
        const problem* prob;
        
    public:
        Model(const problem* prob) : prob{prob} {};
        Model(const char* model_file_name) : prob{nullptr}
        {
            std::cout << "Loading model from " << model_file_name << "...\n";
            this->load(model_file_name);
        };

        virtual void printStruct();
        virtual void printInfo(const bool verbose = 0);
        virtual void performCrossValidation(unsigned int k);
        virtual void reset();
        virtual void fit(const std::vector<unsigned long>& ign_index = {}, const bool verbose = 1);
        virtual unsigned long predict(const feature_node* x);
        virtual double predict_proba(const feature_node* x, const std::vector<unsigned long> ind = {});
        virtual unsigned long getNrClass();
        virtual unsigned long getNrFeatures();
        virtual void save(const char* model_file_name);
        virtual void load(const char* model_file_name); /* TODO: include error handling! */
};

#endif