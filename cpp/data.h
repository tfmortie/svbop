/*
Author: Thomas Mortier 2019

Data processor
*/

#ifndef DATA_U
#define DATA_U

#include "arg.h"
#include "liblinear/linear.h"

void getProblem(ParseResult &presult, problem &p);
int getSizeData(const std::string &file);
void processData(const std::string &file, problem &p);
void processHierarchy(const std:: string &file);


#endif