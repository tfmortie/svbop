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
    ParseResult parser_result {true, "", "./model.out", "", ModelType::L1_LR_PRIMAL, -1, 0, 0.0, 0.0, 0, 0.0};
    parseArgs(argc, argv, parser_result);
    if (parser_result.train)
    {
        // process data and create problem instance
        problem prob;
        getProblem(parser_result, prob);
        // set parameter object
        parameter param;
        // first model specific params
        if (parser_result.model_type != ModelType::HS)
        {
            param.C = parser_result.C;
            param.eps = parser_result.eps;
            param.solver_type = (parser_result.model_type == ModelType::L1_LR_PRIMAL ? L1R_LR : L2R_LR_DUAL);
        }
        else
        {
            param.ne = parser_result.ne;
            param.lr = parser_result.lr;
        }
        // TODO: weighting (imbalanced classification problems) not supported yet!
        param.nr_weight = 0;
        param.weight_label = NULL;
        param.weight = NULL;
        param.p = 0; 
        param.init_sol = NULL;

        // check if flat or hierarchical model
        if (parser_result.model_type != ModelType::HS)
        {
            // create model
            FlatModel model = FlatModel(&prob, &param);
            // check param
            model.printInfo();
            model.performCrossValidation();
            // train model on complete dataset
            model.fit();
            std::cout << "[info] Number of classes in model: " << model.getNrClass() << '\n';
            model.save(parser_result.model_path.c_str());
        }
        else
        {
            HierModel model = HierModel(&prob, &param);
            model.printInfo(0); // TODO: remove (debug)
            model.performCrossValidation(2);
            model.fit();
            model.save(parser_result.model_path.c_str());
        }
        destroy_param(&param);
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
            HierModel model = HierModel(parser_result.model_path.c_str());
            model.printInfo(0);
            double acc {0.0};
            double n_cntr {0.0};
            for(unsigned int n=0; n<static_cast<unsigned int>(prob.l); ++n)
            {
                double pred {model.predict(prob.x[n])};
                double targ {prob.y[n]};
                acc += (pred==targ);
                n_cntr += 1.0;
            }
            std::cout << "Test accuracy: " << (acc/n_cntr)*100.0 << "% \n";
        }
    }
    return 0;
}