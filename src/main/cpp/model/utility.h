/* 
    Author: Thomas Mortier 2019-2020

    Header for working with set-based utilities
*/

#ifndef UTILITY_H
#define UTILITY_H

#include <vector>

enum class UtilityType {
    /* Del Coz et al., 2009 */
    PRECISION, 
    RECALL, 
    FB, 
    /* Zaffalon et al., 2012 */
    CREDAL, 
    EXP, 
    LOG, 
    /* Ramaswamy et al., 2015 */
    REJECT, 
    /* Mortier et al., 2019 */
    GENREJECT 
};

struct param
{
    UtilityType utility {UtilityType::GENREJECT};
    double beta {1.0};
    double delta {1.6};
    double gamma {0.6};
    double alpha {1.0};
    unsigned long K {0};
};

double g(const std::vector<unsigned long>& pred, const param& params);
double u(const std::vector<unsigned long>& pred, unsigned long y, const param& params);
unsigned int parseParamValues(std::string paramvals, param& params);
std::string toStr(UtilityType type);

#endif