/* 
    Author: Thomas Mortier 2019-2020

    Header data processor 
*/

#ifndef DATA_U
#define DATA_U

#include "arg.h"
#include "model/model.h"

inline const std::string ARRDELIM {"],["}; /* seperator for extracting hierarchy */

void getProblem(ParseResult &presult, problem &p);
unsigned long getSizeData(const std::string &file);
void processData(const std::string &file, problem &p);
std::vector<std::vector<unsigned long>> processHierarchy(const std::string &file);
std::vector<std::vector<unsigned long>> strToHierarchy(std::string str);
std::vector<unsigned long> arrToVec(const std::string &s);
std::string vecToArr(const std::vector<unsigned long> &v);

#endif