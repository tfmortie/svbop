/* 
    Author: Thomas Mortier 2019-2020

    Header for working with set-based utilities
*/

#ifndef UTILITY_H
#define UTILITY_H

#include <vector>

enum class UtilityType {
    PRECISION, /* Del Coz et al., 2009 */
    RECALL, /* Del Coz et al., 2009 */
    FB, /* Del Coz et al., 2009 */
    CREDAL, /* Zaffalon et al., 2012 */
    EXP, /* Zaffalon et al., 2012 */
    LOG, /* Zaffalon et al., 2012 */
    REJECT, /* Ramaswamy et al., 2015 */
    GENREJECT /* Mortier et al., 2019 */
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

double g(std::vector<unsigned long> pred, param params);
double u(std::vector<unsigned long> pred, unsigned long y, param params);

#endif