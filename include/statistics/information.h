
#ifndef INFORMATION_H_
#define INFORMATION_H_

#include <statistics/distributions/distributions.h>
#include <statistics/distributions/gmm.h>
#include <queue>

class Smooth{

public:

    Smooth();

    double inline exponential(double x, double alpha){
        if(bFirstExp){
            s = x;
        }else{
            s = alpha * x  + (1 - alpha) * s;
        }

        return s;
    }

    void reset(){
        bFirstExp=true;
        s = 0;
    }


private:

    double s;
    bool bFirstExp;


};

class Information{

public:

    Information();

    double monte_carlo_entropy(const GMM& gmm);

    double lower_bound_entropy(GMM &gmm);

    double upper_bound_entropy(GMM& gmm);

    double entropy_gmm_one_gaussian(const arma::mat& covariance);

    double scaled_sum_eigenvalue(const arma::mat& covariance,double min, double max);

    double sum_eigevalues(const arma::mat& covariance);

private:

    double checkForNan(double x,std::deque<double>& q);

    inline  double rescale(double x,double min,double max, double a, double b) const{
        return (b-a) * (x - min) / (max - min) + a;
    }


private:

    std::deque<double> qHl,qHu,qHo;
};

#endif
