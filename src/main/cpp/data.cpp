/* Author: Thomas Mortier 2019-2020

   Data processor
*/

#include "data.h"
#include <string>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>
#include <iterator>

/* set problem based on parsed arguments */
void getProblem(ParseResult &presult, problem &p)
{
    // get number of observations and features (+ bias)
    p.n = getSizeData(presult.file_path);
    p.d = presult.num_features + ((p.bias > -1) ? 1 : 0);
    // add structure classification problem 
    p.hstruct = processHierarchy(presult.hierarchy_path);
    // get x (features) and y (labels)
    p.y = new unsigned long[p.n];
    p.X = new feature_node*[p.n];
    // and bias
    p.bias = presult.bias;
    processData(presult.file_path, p);
    // and number of epochs and learning rate 
    p.ne = presult.ne;
    p.lr = presult.lr;
}

/* get size of data */
unsigned long getSizeData(const std::string &file)
{
    unsigned long n {0};
    std::ifstream in {file};
    std::string buf;
    while (std::getline(in, buf))
        ++n;
    return n;
}

/* read and write data in problem object */
void processData(const std::string &file, problem &p)
{
    std::ifstream in {file};
    std::string line;
    unsigned long i {0};
    try
    {
        while (std::getline(in, line))
        {
            // get tokens for line (ie class and index:ftval)
            std::istringstream istr_stream {line};
            std::vector<std::string> tokens {std::istream_iterator<std::string> {istr_stream}, std::istream_iterator<std::string>{}};
            // we can now init our feature_node row
            if (p.bias > -1)
                p.X[i] = new feature_node[static_cast<unsigned long>(tokens.size()+1)]; // +1 for bias and terminator
            else
                p.X[i] = new feature_node[static_cast<unsigned long>(tokens.size())]; 
            // assign class 
            p.y[i] = std::stol(tokens[0]);
            for (unsigned int j=1; j<tokens.size(); ++j)
            {
                try 
                {
                     // token represents int:double, hence split on :
                    std::string str_int {tokens[j].substr(0, tokens[j].find(":"))};
                    std::string str_double {tokens[j].substr(tokens[j].find(":")+1, std::string::npos)};
                    // now assign to feature_node
                    p.X[i][j-1].index = std::stol(str_int)+1;
                    p.X[i][j-1].value = std::stod(str_double);  
                }
                catch( std::exception& e)
                {
                    std::cerr << "[error] Exception " << e.what() << " catched!\n";
                    exit(1);
                }
            }   
            // add bias if needed
            if (p.bias > -1)
            {
                p.X[i][tokens.size()-1].index = p.d;
                p.X[i][tokens.size()-1].value = p.bias;
                p.X[i][tokens.size()].index = -1;
                p.X[i][tokens.size()].value = 0.0;
            }
            // add terminator
            unsigned long offset {static_cast<unsigned long>((p.bias > -1 ? 0 : -1))};
            p.X[i][tokens.size()+offset].index = -1;
            p.X[i][tokens.size()+offset].value = 0.0;
            ++i;
        }
    }
    catch(std::ifstream::failure e)
    {
        std::cerr << "[error] Exception " << e.what() << " catched!\n";
        exit(1);
    }
}

/* read and process hierarchy from file */
std::vector<std::vector<unsigned long>> processHierarchy(const std:: string &file)
{
    // first check if file only consists of one line 
    if (getSizeData(file) > 1)
    {
        std::cerr << "[error] Hierarchy file is not in correct format!\n";
        exit(1); 
    }
    std::ifstream in {file};
    std::string line;
    std::vector<std::vector<unsigned long>> h_struct;
    try
    {
        while (std::getline(in, line))
            h_struct = strToHierarchy(line);
    }
    catch(std::ifstream::failure e)
    {
        std::cerr << "[error] Exception " << e.what() << " catched!\n";
        exit(1);
    }
    return h_struct;
}

/* convert string representation of hierarchy to nested vector representation */
std::vector<std::vector<unsigned long>> strToHierarchy(std::string str)
{
    std::vector<std::vector<unsigned long>> h_struct;
    // remove leading and trailing garbage
    str = str.substr(2, str.length()-4);
    // as long as we have more than one index array continue
    while (str.find(ARRDELIM) != std::string::npos)
    {
        auto delim_loc {str.find(ARRDELIM)};
        // get string representation for array
        std::string temp_arr_str {str.substr(0, delim_loc)};
        // convert to vector 
        h_struct.push_back(arrToVec(temp_arr_str));
        // get remaining string
        str = str.substr(delim_loc+3, str.length()-delim_loc-ARRDELIM.length());
    }
    // make sure to process the last bit of our string and we are done
    h_struct.push_back(arrToVec(str));
    return h_struct;
}

/* convert string of format {[0-9][0-9]*,}*[0-9] to vector */
std::vector<unsigned long> arrToVec(const std::string &s)
{
    std::string const DELIM {","};
    std::vector<unsigned long> ret_vec {};
    std::string substr {s};
    while (substr.find(DELIM) != std::string::npos)
    {
        auto delim_loc {substr.find(DELIM)};
        ret_vec.push_back(std::stol(substr.substr(0, delim_loc)));
        substr = substr.substr(delim_loc+1, substr.length()-delim_loc-DELIM.length());
    }
    // make sure to process the last bit of our string
    ret_vec.push_back(std::stol(substr));
    return ret_vec;
}

/* convert vector to string with format [{[0-9][0-9]*,}*[0-9]] */
std::string vecToArr(const std::vector<unsigned long> &v)
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