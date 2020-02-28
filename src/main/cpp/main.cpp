/* 
    Author: Thomas Mortier 2019-2020

    Main

    TODO: early stopping
    TODO: add adagrad/adam optimization
    TODO: add support for mini-batch training
    TODO: improve argument checking
    TODO: check for mem leaks
*/

#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include "arg.h"
#include "data.h"
#include "model/model.h"
#include "model/flat.h"
#include "model/hierarchical.h"
#include "model/utility.h"

/*  main call */
int main(int argc, char** argv)
{
    ParseResult parser_result;
    parseArgs(argc, argv, parser_result);
    if (parser_result.train)
    {
        // process data and create problem instance
        problem prob;
        getProblem(parser_result, prob);
        // check if flat or hierarchical model
        Model* model;
        if (parser_result.model_type == ModelType::SOFTMAX)
            model = new FlatModel(&prob);
        else
            model = new HierModel(&prob);
        model->printInfo(0); 
        //model.performCrossValidation(2);
        // train model 
        // first create index vector for holdout set (used for validation)
        std::vector<unsigned long> ind_data;
        for(unsigned long i=0; i<prob.n; ++i)
            ind_data.push_back(i);
        // now shuffle index vector 
        auto rng = std::default_random_engine {};
        rng.seed(parser_result.seed);
        std::shuffle(std::begin(ind_data), std::end(ind_data), rng);
        // extract first observations, depending on holdout ratio, for holdout set
        std::vector<unsigned long>::const_iterator i_start = ind_data.begin();
        std::vector<unsigned long>::const_iterator i_stop = ind_data.begin() + static_cast<long>(parser_result.holdout*prob.n);
        std::vector<unsigned long> ind_holdout(i_start, i_stop);
        auto t1 = std::chrono::high_resolution_clock::now();
        model->fit(ind_holdout);
        auto t2 = std::chrono::high_resolution_clock::now();
        auto time = std::chrono::duration_cast <std::chrono::milliseconds>(t2 - t1).count();
        std::cout << "Execution time: " << time << " ms\n";
        model->save(parser_result.model_path.c_str());
        delete model;
    }
    else
    {
        // process data and create problem instance
        problem prob;
        getProblem(parser_result, prob);
        // check if flat or hierarchical model
        Model* model;
        if (parser_result.model_type == ModelType::SOFTMAX)
            model = new FlatModel(parser_result.model_path.c_str(), &prob);
        else
            model = new HierModel(parser_result.model_path.c_str(), &prob);

        model->printInfo(0);
        // predictions (accuracy)
        std::cout << "PROB. MODEL\n";
        auto t1 = std::chrono::high_resolution_clock::now();
        double acc {0.0};
        double U {0.0};
        double setsize {0.0};
        double n_cntr {0.0};
        for(unsigned long n=0; n<prob.n; ++n)
        {
            unsigned long pred {model->predict(prob.X[n])};  
            unsigned long targ {prob.y[n]};
            acc += (pred==targ);
            n_cntr += 1.0;
        }
        std::cout << "Acc.: " << (acc/n_cntr)*100.0 << "%\n";
        auto t2 = std::chrono::high_resolution_clock::now();
        auto time = std::chrono::duration_cast <std::chrono::milliseconds>(t2 - t1).count();
        std::cout << "t: " << time << " ms\n";
        // predictions (UBOP)
        std::cout << "UBOP " << toStr(prob.utility.utility) << std::endl;
        t1 = std::chrono::high_resolution_clock::now();
        acc = 0.0;
        U = 0.0;
        setsize = 0.0;
        n_cntr = 0.0;
        for(unsigned long n=0; n<prob.n; ++n)
        {
            std::vector<unsigned long> pred {model->predict_ubop(prob.X[n])};
            unsigned long targ {prob.y[n]};
            setsize += pred.size();
            acc += u(pred, targ, {UtilityType::RECALL});
            U += u(pred, targ, prob.utility);
            n_cntr += 1.0;
        }
        std::cout << "U: " << (acc/n_cntr)*100.0 << "\n";
        std::cout << "R: " << (U/n_cntr)*100.0 << "\n";
        std::cout << "|Y|: " << (setsize/n_cntr) << '\n';
        t2 = std::chrono::high_resolution_clock::now();
        time = std::chrono::duration_cast <std::chrono::milliseconds>(t2 - t1).count();
        std::cout << "t: " << time << " ms\n";
        // predictions (RBOP)
        std::cout << "RBOP " << toStr(prob.utility.utility) << std::endl;
        t1 = std::chrono::high_resolution_clock::now();
        acc = 0.0;
        U = 0.0;
        setsize = 0.0;
        n_cntr = 0.0;
        for(unsigned long n=0; n<prob.n; ++n)
        {
            std::vector<unsigned long> pred {model->predict_rbop(prob.X[n])};
            unsigned long targ {prob.y[n]};
            setsize += pred.size();
            acc += u(pred, targ, {UtilityType::RECALL});
            U += u(pred, targ, prob.utility);
            n_cntr += 1.0;
        }
        std::cout << "U: " << (acc/n_cntr)*100.0 << "\n";
        std::cout << "R: " << (U/n_cntr)*100.0 << "\n";
        std::cout << "|Y|: " << (setsize/n_cntr) << '\n';
        t2 = std::chrono::high_resolution_clock::now();
        time = std::chrono::duration_cast <std::chrono::milliseconds>(t2 - t1).count();
        std::cout << "t: " << time << " ms\n";
        delete model;
    }
    return 0;
}
