/*
Author: Thomas Mortier 2019

Data processor
*/

#ifndef DATA_U
#define DATA_U

#include "arg.h"
#include "liblinear/linear.h"

inline const std::string ARRDELIM {"],["};

void getProblem(ParseResult &presult, problem &p);
int getSizeData(const std::string &file);
void processData(const std::string &file, problem &p);
std::vector<std::vector<int>> processHierarchy(const std::string &file);
std::vector<std::vector<int>> strToHierarchy(std::string str);
std::vector<int> arrToVec(const std::string &s);
std::string vecToArr(const std::vector<int> &v);


#endif