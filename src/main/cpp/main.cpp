/* 
    Author: Thomas Mortier 2019-2020

    Main file

    TODO: check for mem leaks
    TODO: add support for mini-batch training
    TODO: add adagrad/adam optimization
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

/*  
    Main call 

    TODO: clean and optimize
*/
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
            FlatModel model = FlatModel(parser_result.model_path.c_str(), &prob);
            model.printInfo(0);
            // predictions (accuracy)
            std::cout << "PROB. MODEL\n";
            auto t1 = std::chrono::high_resolution_clock::now();
            double acc {0.0};
            double U {0.0};
            double setsize {0.0};
            double n_cntr {0.0};
            for(unsigned long n=0; n<prob.n; ++n)
            {
                unsigned long pred {model.predict(prob.X[n])};  
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
                std::vector<unsigned long> pred {model.predict_ubop(prob.X[n])};
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
                std::vector<unsigned long> pred {model.predict_rbop(prob.X[n])};
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
        }
        else
        {
            HierModel model = HierModel(parser_result.model_path.c_str(), &prob);
            model.printInfo(0);
            // predictions (accuracy)
            std::cout << "PROB. MODEL\n";
            auto t1 = std::chrono::high_resolution_clock::now();
            double acc {0.0};
            double U {0.0};
            double setsize {0.0};
            double n_cntr {0.0};
            for(unsigned long n=0; n<prob.n; ++n)
            {
                unsigned long pred {model.predict(prob.X[n])};  
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
                std::vector<unsigned long> pred {model.predict_ubop(prob.X[n])};
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
                std::vector<unsigned long> pred {model.predict_rbop(prob.X[n])};
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
        }
    }
    return 0;
}
