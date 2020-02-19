/* 
    Author: Thomas Mortier 2019-2020

    Main file

    TODO: add support for mini-batch training
*/

#include <iostream>
#include <vector>
#include <chrono>
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
        if (parser_result.model_type == ModelType::SOFTMAX)
        {
            FlatModel model = FlatModel(&prob);
            model.printInfo(0); 
            //model.performCrossValidation(2);
            // train model on complete dataset
            auto t1 = std::chrono::high_resolution_clock::now();
            model.fit();
            auto t2 = std::chrono::high_resolution_clock::now();
            auto time = std::chrono::duration_cast <std::chrono::milliseconds>(t2 - t1).count();
            std::cout << "Execution time: " << time << " ms\n";
            model.save(parser_result.model_path.c_str());
        }
        else
        {
            HierModel model = HierModel(&prob);
            model.printInfo(0); 
            //model.performCrossValidation(2);
            // train model on complete dataset
            auto t1 = std::chrono::high_resolution_clock::now();
            model.fit();
            auto t2 = std::chrono::high_resolution_clock::now();
            auto time = std::chrono::duration_cast <std::chrono::milliseconds>(t2 - t1).count();
            std::cout << "Execution time: " << time << " ms\n";
            model.save(parser_result.model_path.c_str());
        }
    }
    else
    {
        // process data and create problem instance
        problem prob;
        getProblem(parser_result, prob);
        // check if flat or hierarchical model
        if (parser_result.model_type == ModelType::SOFTMAX)
        {
            FlatModel model = FlatModel(parser_result.model_path.c_str());
            model.printInfo(0);
            auto t1 = std::chrono::high_resolution_clock::now();
            double acc {0.0};
            double n_cntr {0.0};
            for(unsigned long n=0; n<prob.n; ++n)
            {
                unsigned long pred {model.predict(prob.X[n])};  
                unsigned long targ {prob.y[n]};
                acc += (pred==targ);
                n_cntr += 1.0;
            }
            std::cout << "Test accuracy: " << (acc/n_cntr)*100.0 << "%\n";
            auto t2 = std::chrono::high_resolution_clock::now();
            auto time = std::chrono::duration_cast <std::chrono::milliseconds>(t2 - t1).count();
            std::cout << "Execution time: " << time << " ms\n";
        }
        else
        {
            HierModel model = HierModel(parser_result.model_path.c_str());
            model.printInfo(0);
            auto t1 = std::chrono::high_resolution_clock::now();
            double acc {0.0};
            double n_cntr {0.0};
            for(unsigned long n=0; n<prob.n; ++n)
            {
                unsigned long pred {model.predict(prob.X[n])};
                unsigned long targ {prob.y[n]};
                acc += (pred==targ);
                n_cntr += 1.0;
            }
            std::cout << "Test accuracy: " << (acc/n_cntr)*100.0 << "%\n";
            auto t2 = std::chrono::high_resolution_clock::now();
            auto time = std::chrono::duration_cast <std::chrono::milliseconds>(t2 - t1).count();
            std::cout << "Execution time: " << time << " ms\n";
        }
    }
    return 0;
}
