/* 
    Author: Thomas Mortier 2019-2020

    Set-based utilities implementations
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