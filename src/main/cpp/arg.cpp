/* 
    Author: Thomas Mortier 2019-2020

    Argument parser
*/

#include <iostream>
#include <string>
#include "arg.h"
#include "model/utility.h"

/* print help information to stderr */
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
                0 := softmax with SGD
                1 := hierarchical softmax with SGD and slow training
                2 := hierarchical sofmax with SGD and fast training
        -s, --struct            Structure classification problem
        -b, --bias              Bias for linear model 
              >=0 := bias included 
              <0  := bias not included 
        -ne, --nepochs          Number of epochs
        -lr, --learnrate        Learning rate 
        -d, --dim               Number of features of dataset (bias not included)
        -u, --utility           Utility function (format: {precision|recall|fb|credal|exp|log|reject|genreject})
        -p, --param             Parameters for utility (format: [valparam1,valparam2,...])
        -m, --model             Model path for predicting/saving
        -f, --file              File path for saving predictions (if specified)           
    )help"; 
    exit(1);
}

/* parse arguments provided to main call */
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
                presult.model_type = ModelType::SOFTMAX;
            else if (static_cast<std::string>(args[i+1]).compare("1") == 0)
                presult.model_type = ModelType::HSOFTMAXS;
            else if (static_cast<std::string>(args[i+1]).compare("2") == 0)
                presult.model_type = ModelType::HSOFTMAXF;
            else
            {
                std::cerr << "[error] Model type " << static_cast<std::string>(args[i+1]) << " not defined!\n";
                showHelp();
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
        else if (static_cast<std::string>(args[i]).compare("-s") == 0 || static_cast<std::string>(args[i]).compare("--struct") == 0)
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
            presult.num_features = static_cast<unsigned long>(std::stol(args[i+1]));
            ++i;
        }
        // check for -ne, --nepochs
        else if (static_cast<std::string>(args[i]).compare("-ne") == 0 || static_cast<std::string>(args[i]).compare("--nepochs") == 0)
        {
            presult.ne = static_cast<unsigned int>(std::stoi(args[i+1]));
            ++i;
        }
        // check for -lr, --learnrate
        else if (static_cast<std::string>(args[i]).compare("-lr") == 0 || static_cast<std::string>(args[i]).compare("--learnrate") == 0)
        {
            presult.lr = std::stod(args[i+1]);
            ++i;
        }
        // check for -u, --utility
        else if (static_cast<std::string>(args[i]).compare("-u") == 0 || static_cast<std::string>(args[i]).compare("--utility") == 0)
        {
            // process value for argument -u 
            if (static_cast<std::string>(args[i+1]).compare("precision") == 0)
                presult.utility_params.utility = UtilityType::PRECISION;
            else if (static_cast<std::string>(args[i+1]).compare("recall") == 0)
                presult.utility_params.utility = UtilityType::RECALL;
            else if (static_cast<std::string>(args[i+1]).compare("fb") == 0)
                presult.utility_params.utility = UtilityType::FB;
            else if (static_cast<std::string>(args[i+1]).compare("credal") == 0)
                presult.utility_params.utility = UtilityType::CREDAL;
            else if (static_cast<std::string>(args[i+1]).compare("exp") == 0)
                presult.utility_params.utility = UtilityType::EXP;
            else if (static_cast<std::string>(args[i+1]).compare("log") == 0)
                presult.utility_params.utility = UtilityType::LOG;
            else if (static_cast<std::string>(args[i+1]).compare("reject") == 0)
                presult.utility_params.utility = UtilityType::REJECT;
            else if (static_cast<std::string>(args[i+1]).compare("genreject") == 0)
                presult.utility_params.utility = UtilityType::GENREJECT;
            else
            {
                std::cerr << "[error] Utility type " << static_cast<std::string>(args[i+1]) << " not defined!\n";
                showHelp();
            }
            ++i;
        }
        // check for -p, --param
        else if (static_cast<std::string>(args[i]).compare("-p") == 0 || static_cast<std::string>(args[i]).compare("--param") == 0)
        {
            // process value for argument -p 
            unsigned int ret {parseParamValues(static_cast<std::string>(args[i+1]), presult.utility_params)};
            if (ret!=0)
            {
                std::cerr << "[error] Values " << static_cast<std::string>(args[i+1]) << " for specified utility not correct !\n";
                showHelp();
            }
            ++i;
        }
        // check for -f, --file
        else if (static_cast<std::string>(args[i]).compare("-f") == 0 || static_cast<std::string>(args[i]).compare("--file") == 0)
        {
            presult.pred_path= args[i+1];
            ++i;
        }
        else
        {
            std::cerr << "[error] Argument " << args[i] << " not defined!\n";
            showHelp();
        }
    }
}

/* 
    Check correctness of provided arguments
    TODO: check if all information relevant for the problem is specified through ParseResult
*/
void checkArgs(const ParseResult& presult)
{
    std::cout << "[info] Not implemented yet!\n";
}