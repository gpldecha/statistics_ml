#ifndef DECISION_FUNCTIONS_H_
#define DECISION_FUNCTIONS_H_

#include <armadillo>

class Decision_functions{

public:

    template<typename T>
    static T gaussian_step_function(T x,const T mean, const T beta){
        if(x > mean){
            return 1.0;
        }else{
            return exp(-beta * (x - mean) * (x - mean));
        }
    }


    static double step_function(const double x, const double threashold){
        if(x > threashold){
            return 1;
        }else{
            return 0;
        }
    }

    static double bound_gaussian_step_function(const double x, const double mean, const double beta, const double up, const double down){
        if( x > up){
            return 1;
        }else if(x < down){
            return 0;
        }else{
            return exp(-beta * (x - mean) * (x - mean));
        }

    }


    static double bound_mv_gaussian_step_function(const arma::colvec& x,arma::colvec& mean, const double beta, const double up, const double down){
        if(arma::dot(x - mean,x - mean) > up){
            return 0;
        }else if(arma::dot(x - mean,x - mean) < down){
            return 1;
        }else{
            return exp(-beta * arma::dot(x - mean,x - mean));
        }
    }



};


#endif
