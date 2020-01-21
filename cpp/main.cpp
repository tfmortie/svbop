/*
Author: Thomas Mortier 2019

Main file
*/

// TODO: finalize comments

#include <iostream>
#include <vector>
#include "arg.h"
#include "data.h"
#include "model/flat.h"
#include "model/hierarchical.h"
#include "liblinear/linear.h"

int main(int argc, char** argv)
{
    ParseResult parser_result {true, "", "./model.out", "", ModelType::L1_LR_PRIMAL, -1, 0, 0.0, 0.0};
    parseArgs(argc, argv, parser_result);
    if (parser_result.train)
    {
        // process data and create problem instance
        problem prob;
        getProblem(parser_result, prob);
        // check if flat or hierarchical model
        if (parser_result.model_type != ModelType::HS)
        {
            // create parameter
            parameter param;
            param.C = parser_result.C;
            param.eps = parser_result.eps;
            param.solver_type = (parser_result.model_type == ModelType::L1_LR_PRIMAL ? L1R_LR : L2R_LR_DUAL);
            // TODO: weighting (imbalanced classification problems) not supported yet!
            param.nr_weight = 0;
            param.weight_label = NULL;
            param.weight = NULL;
            param.p = 0; 
            param.init_sol = NULL;
            // create model
            FlatModel model = FlatModel(&prob, &param);
            // check param
            model.printInfo();
            model.performCrossValidation();
            // train model on complete dataset
            model.fit();
            std::cout << "[info] Number of classes in model: " << model.getNrClass() << '\n';
            model.save(parser_result.model_path.c_str());
            destroy_param(&param);
        }
        else
        {
            HierModel* model = new HierModel(prob);
            model->printInfo(); // TODO: remove (debug)
            model->fit(20, 0.000001);
            // TODO: remove below (debug)
            double acc {0.0};
            for (unsigned int n = 0; n<static_cast<unsigned int>(prob.l); ++n)
            //for (unsigned int n = 0; n<10; ++n)
            {
                double pred {model->predict(prob.x[n])};
                double targ {prob.y[n]};
                std::cout << "Pred: " << pred << '\n';
                std::cout << "Targ: " << targ << '\n';
                acc += (pred==targ);
            }
            std::cout << "Fitting accuracy: " << (acc/prob.l)*100.0 << "% \n";
            //std::cout << "Fitting accuracy: " << (acc/10.0)*100.0 << '\n';
            delete model;
        }
    }
    else
    {
        // process data and create problem instance
        problem prob;
        getProblem(parser_result, prob);
        // check if flat or hierarchical model
        if (parser_result.model_type != ModelType::HS)
        {
            FlatModel model = FlatModel(parser_result.model_path.c_str());
            double* prob_estimates = new double[static_cast<unsigned long>(model.getNrClass())];
            double cls_pred {model.predict(prob.x[0])};
            std::cout << "Predicted class: " << cls_pred << '\n';
            model.predict_proba(prob.x[0], prob_estimates);
            for (unsigned int i = 0; i<static_cast<unsigned int>(model.getNrClass()); ++i)
            {
                std::cout << "Class " << i << " with prob: " << prob_estimates[i] << '\n';
            }
            delete[] prob_estimates;
        }
        else
        {
            std::cerr << "[error] Not implemented yet!";
            exit(1);
        }
    }
    return 0;
}

