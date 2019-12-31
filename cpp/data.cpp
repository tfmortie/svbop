/*
Author: Thomas Mortier 2019

Data processor
*/

#include "data.h"
#include <string>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>
#include <iterator>

void getProblem(ParseResult &presult, problem &p)
{
    // initialize vars that are already determined
    p.bias = presult.bias;
    p.n = presult.num_features + ((p.bias > -1) ? 1 : 0);
    // get number of observations 
    p.l = getSizeData(presult.file_path);
    // get x (features) and y (labels)
    p.y = new double[static_cast<unsigned long>(p.l)];
    p.x = new feature_node*[static_cast<unsigned long>(p.l)];
    processData(presult.file_path, p);
}

int getSizeData(const std::string &file)
{
    int n {0};
    std::ifstream in {file};
    std::string buf;
    while (std::getline(in, buf))
        ++n;
    return n;
}

void processData(const std::string &file, problem &p)
{
    std::ifstream in {file};
    std::string line;
    unsigned int i {0};
    try
    {
        while (std::getline(in, line))
        {
            // get tokens for line (ie class and index:ftval)
            std::istringstream istr_stream {line};
            std::vector<std::string> tokens {std::istream_iterator<std::string> {istr_stream}, std::istream_iterator<std::string>{}};
            // we can now init our feature_node row
            if (p.bias > -1)
                p.x[i] = new feature_node[static_cast<unsigned long>(tokens.size()+1)]; // +1 for bias and terminator
            else
                p.x[i] = new feature_node[static_cast<unsigned long>(tokens.size())]; 
            // assign class 
            p.y[i] = std::stod(tokens[0]);
            for (unsigned int j=1; j<tokens.size(); ++j)
            {
                try 
                {
                     // token represents int:double, hence split on :
                    std::string str_int {tokens[j].substr(0, tokens[j].find(":"))};
                    std::string str_double {tokens[j].substr(tokens[j].find(":")+1, std::string::npos)};
                    // now assign to feature_node
                    p.x[i][j-1].index = std::stoi(str_int)+1;
                    p.x[i][j-1].value = std::stold(str_double);  
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
                p.x[i][tokens.size()-1].index = p.n;
                p.x[i][tokens.size()-1].value = p.bias;
                p.x[i][tokens.size()].index = -1;
                p.x[i][tokens.size()].value = 0.0;
            }
            // add terminator
            unsigned long offset {static_cast<unsigned long>((p.bias > -1 ? 0 : -1))};
            p.x[i][tokens.size()+offset].index = -1;
            p.x[i][tokens.size()+offset].value = 0.0;
            ++i;
        }
    }
    catch(std::ifstream::failure e)
    {
        std::cerr << "[error] Exception " << e.what() << " catched!\n";
        exit(1);
    }
}

// TODO
void processHierarchy(const std:: string &file)
{
    // first check if file only consists of one line 
    if (getSizeData(file) > 1)
    {
        std::cerr << "[error] Hierarchy file is not in correct format!\n";
        exit(1); 
    }
    std::ifstream in {file};
    std::string line;
    unsigned int i {0};
    try
    {
        while (std::getline(in, line))
        {
            
        }
    }
    catch(std::ifstream::failure e)
    {
        std::cerr << "[error] Exception " << e.what() << " catched!\n";
        exit(1);
    }
}
