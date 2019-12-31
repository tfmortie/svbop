/*
Author: Thomas Mortier 2019

Argument parser
*/

#include <iostream>
#include <string>
#include "arg.h"

void showHelp()
{
    std::cerr << R"help(  
    HELP: svp <command> <args>

    command:
        train                   Training mode
        predict                 Predict mode
        -h, --help              Help documentation
    
    args:
        -i, --input             Training/prediction data in LIBSVM format
        -t, --type              Model type for training
                0 := L1-regularized logistic regression
                1 := L2-regularized logistic regression (dual)
        -m, --model             Model path for predicting/saving
        -h, --hierarchy         Hierarchy path for h-softmax
        -b, --bias              Bias for linear model 
              >=0 := bias included 
              <0  := bias not included 
        -C, --C                 The cost of constraints violation
        -e, --eps               Stopping criteria
        -d, --dim               Number of features of dataset (bias not included)
    )help"; 
    exit(1);
}

void parseArgs(int argc, char** args, ParseResult& presult)
{
    // commands/arguments specified?
    if (argc == 1)
    {
        std::cerr << "[error] No commands/arguments specified!\n";
        showHelp();
    }
    std::string arg_command {static_cast<std::string>(args[1])};
    if (arg_command.compare("train") == 0)
    {
        // train command used
        presult.train = true;
    }
    else if (arg_command.compare("predict") == 0)
    {
        // predict command used
        presult.train = false;
    }
    else if (arg_command.compare("-h") == 0 || arg_command.compare("--help") == 0)
    {
        showHelp();
    }
    else
    {
        std::cerr << "[error] Command " << arg_command << " not defined!\n";
        showHelp();
    }
    // run over arguments
    for (unsigned int i = 2; i < static_cast<unsigned int>(argc); ++i)
    {
        // check for -i, --input
        if (static_cast<std::string>(args[i]).compare("-i") == 0 || static_cast<std::string>(args[i]).compare("--input") == 0)
        {
            presult.file_path = args[i+1];
            ++i;
        }
        // check for -t, --type
        else if (static_cast<std::string>(args[i]).compare("-t") == 0 || static_cast<std::string>(args[i]).compare("--type") == 0)
        {
            // process value for argument -t
            if (static_cast<std::string>(args[i+1]).compare("0") == 0)
            {
                presult.model_type = ModelType::L1_LR_PRIMAL;
            }
            else
            {
                presult.model_type = ModelType::L1_LR_DUAL;
            }
            ++i;
        }
        // check for -m, --model
        else if (static_cast<std::string>(args[i]).compare("-m") == 0 || static_cast<std::string>(args[i]).compare("--model") == 0)
        {
            presult.model_path = args[i+1];
            ++i;
        }
        // check for -h, --hierarchy
        else if (static_cast<std::string>(args[i]).compare("-h") == 0 || static_cast<std::string>(args[i]).compare("--hierarchy") == 0)
        {
            presult.hierarchy_path = args[i+1];
            ++i;
        }
        // check for -b, --bias
        else if (static_cast<std::string>(args[i]).compare("-b") == 0 || static_cast<std::string>(args[i]).compare("--bias") == 0)
        {
            presult.bias = std::stod(args[i+1]);
            ++i;
        }
        // check for -d, --dim
        else if (static_cast<std::string>(args[i]).compare("-d") == 0 || static_cast<std::string>(args[i]).compare("--dim") == 0)
        {
            presult.num_features = std::stoi(args[i+1]);
            ++i;
        }
        // check for -C, --C
        else if (static_cast<std::string>(args[i]).compare("-C") == 0 || static_cast<std::string>(args[i]).compare("--C") == 0)
        {
            presult.C = std::stod(args[i+1]);
            ++i;
        }
        // check for -e, --eps
        else if (static_cast<std::string>(args[i]).compare("-e") == 0 || static_cast<std::string>(args[i]).compare("--eps") == 0)
        {
            presult.eps = std::stod(args[i+1]);
            ++i;
        }
        else
        {
            std::cerr << "[error] Argument " << args[i] << " not defined!\n";
            showHelp();
        }
    }
}

// TODO: check if all information relevant for the problem is specified through ParseResult
void checkArgs(const ParseResult& presult)
{
    std::cout << "[info] Not implemented yet!\n";
}