/* Author: Thomas Mortier 2019-2020

   Header argument parser
*/

#ifndef ARG_U
#define ARG_U

#include <string>
#include "model/model.h"

struct ParseResult {
    /* MODE */
    bool train;
    /* FILE PATHS */
    std::string file_path;
    std::string model_path;
    std::string hierarchy_path;
    /* MODEL */
    ModelType model_type;
    double bias; 
    int num_features;
    int ne;
    double lr;
};

void showHelp();
void parseArgs(int argc, char** args, ParseResult& presult);
void checkArgs(const ParseResult& presult);

#endif
