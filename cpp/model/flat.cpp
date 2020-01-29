/*
Author: Thomas Mortier 2019

Implementation of flat model
*/

// TODO: finalize comments

#include "model/flat.h"
#include <iostream>
#include <utility> 

FlatModel::FlatModel(const char* model_file_name) : prob{nullptr}, param{nullptr}, model{load_model(model_file_name)}
{
    std::cout << "Loading model from " << model_file_name << "...\n";
    model = load_model(model_file_name); 
    if (model == nullptr)
    {
        std::cerr << "[error] File "  << model_file_name << " does not exist!\n";
        exit(1);
    }
    // init. class->label dict
    class_to_label_dict = new int[static_cast<unsigned long>(getNrClass())];
    get_labels(model, class_to_label_dict);
}

FlatModel::~FlatModel()
{
    if (model != nullptr)
    {
        free();
    }
}

void FlatModel::printInfo()
{
    if (model != nullptr)
    {
        std::cout << "---------------------------------------------------\n";
        std::cout << "[info] Flat model: \n";
        std::cout << "---------------------------------------------------\n";
        std::cout << "  * Number of features              = " << this->model->nr_feature << '\n';
        std::cout << "  * Number of classes               = " << this->model->nr_class << '\n';
        if (param->solver_type == L1R_LR)
            std::cout << "  * Optimizer                       = L1\n";
        else
            std::cout << "  * Optimizer                       = L2(dual)\n";
        std::cout << "  * C                               = " << this->model->param.C << '\n';
        std::cout << "  * Epsilon                         = " << this->model->param.eps << '\n';
        std::cout << "---------------------------------------------------\n\n";
    }
    else
    {
        std::cerr << "[warning] Model has not been fitted yet!\n";
    }
}

void FlatModel::performCrossValidation()
{
    if (model == nullptr && prob != nullptr && param != nullptr)
    {
        int i;
        int total_correct = 0;
        double *target = new double[static_cast<unsigned long>(prob->l)];
        cross_validation(prob,param,4,target);
        for(i=0;i<prob->l;i++)
	            if(target[i] == prob->y[i])
		            ++total_correct;
        std::cout << "4-fold cross-validation accuracy = " << 100.0*total_correct/prob->l << "% \n";
        delete[] target;
    }
    else
    {
        if(this->model == nullptr)
        {
            std::cerr << "[warning] Model has not been fitted yet!\n";
        }
        else
        {
            std::cerr << "[warning] Model is in predict mode!\n";
        }
    }
}

void FlatModel::fit()
{
    std::cout << "Fit model...\n";
    if (model == nullptr && prob != nullptr && param != nullptr)
    {
        checkParam(); // check param object 
        // train model on complete dataset
        model = {train(prob, param)}; 
        std::cout << "Number of classes in fitted model: " << getNrClass() << '\n';
        // init. class->label dict
        class_to_label_dict = new int[static_cast<unsigned long>(getNrClass())];
        get_labels(model, class_to_label_dict);
    }
    else
    {
        std::cerr << "[warning] Model is in predict mode!\n";
    }
}

// predict class with highest probability 
double FlatModel::predict(const feature_node *x)
{
    if (model == nullptr)
    {
        std::cerr << "[warning] Model has not been fitted yet!\n";
        return -1.0;
    }
    else
    {
        double* prob_estimates = new double[static_cast<unsigned long>(getNrClass())];
        double yhat {predict_probability(model, x, prob_estimates)};
        delete[] prob_estimates;
        return yhat;
    }
}

void FlatModel::predict_proba(const feature_node* x, double* prob_estimates)
{
    if (model == nullptr)
    {
        std::cerr << "[warning] Model has not been fitted yet!\n";
    }
    else
    {
        // fill in probability estimates vector 
        predict_probability(model, x, prob_estimates);
        // and reorder (TODO: perhaps not the most efficient way of reodering)
        int* cls_t_lbl_temp = new int[static_cast<unsigned long>(getNrClass())];
        std::memcpy(cls_t_lbl_temp, class_to_label_dict, static_cast<unsigned long>(getNrClass())*sizeof(int)); 
        for (unsigned int i=0; i<static_cast<unsigned int>(getNrClass()); ++i)
        {
            // switch
            std::swap(prob_estimates[i],prob_estimates[cls_t_lbl_temp[i]-1]);
            std::swap(cls_t_lbl_temp[i],cls_t_lbl_temp[cls_t_lbl_temp[i]-1]);
        }
        delete[] cls_t_lbl_temp;
    }
}
    
// check problem and parameter before training
void FlatModel::checkParam()
{
    const char *error_msg {check_parameter(prob, param)}; // check problem and parameter before training
    if (error_msg)
    {
        std::cerr << "[error] " << error_msg << '\n';
        exit(1);
    }  
}

// get number of classes of fitted model
int FlatModel::getNrClass()
{
    if (model == nullptr)
    {
        std::cerr << "[warning] Model has not been fitted yet!\n";
        return 0;
    }
    else
        return get_nr_class(model);
}

// save model
void FlatModel::save(const char* model_file_name)
{
    std::cout << "Saving model to " << model_file_name << "...\n";
    if (save_model(model_file_name, model))
    {
        std::cerr << "[error] Can't save model to " << model_file_name << "!\n";
        free();
        exit(1);
    }
}

// deallocate model
void FlatModel::free()
{
    delete[] class_to_label_dict;
    free_and_destroy_model(&model);
}