/* 
    Author: Thomas Mortier 2019-2020

    Header data processor 
*/

#ifndef DATA_H
#define DATA_H

#include "arg.h"
#include "model/model.h"

const std::string ARRDELIM {"],["}; /* seperator for extracting hierarchy */

void getProblem(ParseResult &presult, problem &p);
unsigned long getSizeData(const std::string &file);
void processData(const std::string &file, problem &p);
std::vector<std::vector<unsigned long>> processHierarchy(const std::string &file);
std::vector<std::vector<unsigned long>> strToHierarchy(std::string str);
std::vector<unsigned long> arrToVec(const std::string &s);

/* convert vector of arbitrary type to string */
template <typename T> 
std::string vecToArr(const std::vector<T> &v)
{
    std::string ret_arr;
    ret_arr += '[';
    // process all except last element
    for (unsigned int i=0; i<v.size()-1; ++i)
    {
        ret_arr += std::to_string(v[i]);
        ret_arr += ",";
    }
    // and now last element
    ret_arr += std::to_string(v[v.size()-1]);
    ret_arr += "]";
    return ret_arr;
}

#endif