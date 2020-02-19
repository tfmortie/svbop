/* 
    Author: Thomas Mortier 2019-2020

    Set-based utility implementations
*/

#include <iostream>
#include <math.h>
#include <vector>
#include <algorithm>
#include "model/utility.h"

/*
    Calculation of positive utility.

    Arguments:
        pred: prediction which corresponds to a set of labels
        params: information regarding utility
*/
double g(std::vector<unsigned long> pred, param params)
{
    switch(params.utility) {
        case UtilityType::CREDAL: return (params.delta/static_cast<double>(pred.size()))-(params.gamma/pow(static_cast<double>(pred.size()),2.0));
        case UtilityType::EXP: return 1.0-exp(-(params.delta/static_cast<double>(pred.size())));
        case UtilityType::FB: return (1.0+pow(params.beta,2.0))/(static_cast<double>(pred.size())+pow(params.beta,2.0));
        case UtilityType::GENREJECT: return 1.0-params.alpha*pow((static_cast<double>(pred.size())-1.0)/(static_cast<double>(params.K)-1.0),params.beta);
        case UtilityType::LOG: return log(1.0+(1.0/static_cast<double>(pred.size())));
        case UtilityType::PRECISION: return 1.0/static_cast<double>(pred.size());
        case UtilityType::RECALL: return 1.0;
        case UtilityType::REJECT: return 1.0-params.alpha*((static_cast<double>(pred.size())-1.0)/(static_cast<double>(params.K)-1.0));
    default:    std::cerr << "[error] Utility type not recognised!\n";
                return -1.0;
    }
}

/*
    Calculation of utility.

    Arguments:
        pred: prediction which corresponds to a set of labels
        y: ground-truth
        params: information regarding utility
*/
double u(std::vector<unsigned long> pred, unsigned long y, param params)
{
    if (std::find(pred.begin(), pred.end(), y) != pred.end())
        return g(pred, params);
	else
        return 0.0;
}

/*
    Parse string representation of utility parameters.

    Arguments:
        paramvals: utility parameters (format: [paramval1,paramval2,...])
        params: utility parameter struct

    TODO: catch more exceptions
*/
unsigned int parseParamValues(std::string paramvals, param& params)
{
    unsigned int retval {0};
    std::string const DELIM {" "};
    std::vector<double> proc_vec {};
    std::string substr {paramvals};
    while (substr.find(DELIM) != std::string::npos)
    {
        auto delim_loc {substr.find(DELIM)};
        proc_vec.push_back(std::stod(substr.substr(0, delim_loc)));
        substr = substr.substr(delim_loc+1, substr.length()-delim_loc-DELIM.length());
    }
    // make sure to process the last bit of our string
    proc_vec.push_back(std::stod(substr));
    if (params.utility == UtilityType::FB)
        params.beta = proc_vec[0];
    else if (params.utility == UtilityType::CREDAL)
    {
        params.delta = proc_vec[0];
        params.gamma = proc_vec[1];
    }
    else if (params.utility == UtilityType::EXP)
        params.delta = proc_vec[0];
    else if (params.utility == UtilityType::REJECT)
        params.alpha = proc_vec[0];
    else if (params.utility == UtilityType::GENREJECT)
    {
        params.alpha = proc_vec[0];
        params.beta = proc_vec[1];
        params.K = static_cast<unsigned long>(proc_vec[2]);
    }
    else
    {
        retval = 1;
    }
    return retval;
}

/*
    Get string representation for UtilityType.

    Arguments:
        type: utility type
    Return:
        String representation
*/
std::string toStr(UtilityType type)
{
    switch(type) {
        case UtilityType::CREDAL: return "CREDAL";
        case UtilityType::EXP: return "EXP";
        case UtilityType::FB: return "FB";
        case UtilityType::GENREJECT: return "GENREJECT";
        case UtilityType::LOG: return "LOG";
        case UtilityType::PRECISION: return "PRECISION";
        case UtilityType::RECALL: return "RECALL";
        case UtilityType::REJECT: return "REJECT";
    }
}