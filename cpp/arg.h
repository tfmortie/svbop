/*
Author: Thomas Mortier 2019

Argument parser
*/

#ifndef ARG_U
#define ARG_U

#include <string>

enum class ModelType {
    L1_LR_PRIMAL,
    L1_LR_DUAL,
    HS
};

struct ParseResult {
    bool train;
    std::string file_path;
    std::string model_path;
    std::string hierarchy_path;
    ModelType model_type;
    double bias; 
    int num_features;
    double C; // cost of constraints violation
    double eps; // stopping criteria
};

void showHelp();
void parseArgs(int argc, char** args, ParseResult& presult);
void checkArgs(const ParseResult& presult);

#endif
