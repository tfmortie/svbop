/* 
    Author: Thomas Mortier 2019-2020

    Header argument parser
*/

#ifndef ARG_H
#define ARG_H

#include <string>
#include "model/model.h"
#include "model/utility.h"

struct ParseResult {
    /* MODE */
    bool train {true};
    /* FILE PATHS */
    std::string file_path {""};
    std::string model_path {"./model.out"};
    std::string hierarchy_path {""};
    std::string pred_filename {""};
    /* MODEL */
    ModelType model_type {ModelType::SOFTMAX};
    OptimType optim {OptimType::SGD};
    double bias {-1.0}; 
    unsigned long num_features {0};
    unsigned int ne {0};
    double lr {0.0};
    unsigned int batchsize {32};
    double holdout {0.2};
    int patience {5};
    /* UTILITY */ 
    param utility_params;
    /* OTHER */
    int seed {2020};
};

void showHelp();
void parseArgs(int argc, char** args, ParseResult& presult);
void checkArgs(const ParseResult& presult);

#endif
