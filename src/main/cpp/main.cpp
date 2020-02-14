/* 
    Author: Thomas Mortier 2019-2020

    Main file
*/

#include <iostream>
#include <vector>
#include <ctime> 
#include "arg.h"
#include "data.h"
#include "model/model.h"
#include "model/flat.h"
#include "model/hierarchical.h"

/*  main call */
int main(int argc, char** argv)
{
    ParseResult parser_result {true, "", "./model.out", "", ModelType::SOFTMAX, -1.0, 0, 0, 0.0};
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
            model.performCrossValidation(2);
            // train model on complete dataset
            model.fit();
            model.save(parser_result.model_path.c_str());
        }
        else
        {
            HierModel model = HierModel(&prob);
            model.printInfo(0); 
            model.performCrossValidation(2);
            // train model on complete dataset
            model.fit();
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
            time_t start,end; 
            time(&start);
            double acc {0.0};
            double n_cntr {0.0};
            for(unsigned long n=0; n<prob.n; ++n)
            {
                unsigned long pred {model.predict(prob.X[n])};  
                unsigned long targ {prob.y[n]};
                acc += (pred==targ);
                n_cntr += 1.0;
            }
            std::cout << "Test accuracy: " << (acc/n_cntr)*100.0 << "% \n";
            time(&end);
            double dt = difftime(end,start);
            std::cout << "Execution time: " << dt << " s\n";
        }
        else
        {
            HierModel model = HierModel(parser_result.model_path.c_str());
            model.printInfo(0);
            time_t start,end; 
            time(&start);
            double acc {0.0};
            double n_cntr {0.0};
            for(unsigned long n=0; n<prob.n; ++n)
            {
                unsigned long pred {model.predict(prob.X[n])};
                unsigned long targ {prob.y[n]};
                acc += (pred==targ);
                n_cntr += 1.0;
            }
            std::cout << "Test accuracy: " << (acc/n_cntr)*100.0 << "% \n";
            time(&end);
            double dt = difftime(end,start);
            std::cout << "Execution time: " << dt << " s\n";
        }
    }
    return 0;
}