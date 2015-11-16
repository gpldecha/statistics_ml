#ifndef GAUSSIAN_H_
#define GAUSSIAN_H_

// Boost
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/normal_distribution.hpp>
#include <boost/random/uniform_real.hpp>
#include <boost/random/variate_generator.hpp>
#include <boost/random/discrete_distribution.hpp>
#include <boost/range.hpp>
#include <boost/array.hpp>

// Armadillo

#include <armadillo>

namespace stats{

class Gaussian{

    typedef boost::mt19937 ENG;
    typedef boost::normal_distribution<double> DIST;

public:

    Gaussian();

    Gaussian(const arma::colvec& Mean, const arma::mat& Covariance);

    double likelihood(const arma::colvec& X);

    void fit(const arma::mat& data);

    arma::vec sample();

public:

    arma::colvec Mean;
    arma::mat    Covariance;
    arma::mat    invCovariance;
    double       det;

private:

    std::size_t                          D;
    arma::mat                            A; //Cholesky of Cov
    arma::vec                            z;
    boost::mt19937                       generator;
    static boost::mt19937                gens;
    boost::variate_generator<ENG,DIST>   gen;
    boost::normal_distribution<double>   normal;
    double                               denom;


};


/**
 *  Conditional Gaussian function
 *
 *
 *
 *
 **/


class Gaussian_c{

public:

    void condition(const arma::colvec& Mean, const arma::mat& Covariance, const std::vector<std::size_t>& in, const std::vector<std::size_t>& out);

    void mu_condition(const arma::colvec &x);

private:

    void getBlock(arma::mat& A_xx,const arma::mat& A,const std::vector<std::size_t>& dim1,const std::vector<std::size_t>& dim2);

    void getBlock(arma::vec &Mu_xx,const arma::colvec& Mu,const std::vector<std::size_t>& dim);

public:

    arma::mat    Sigma_1c2;
    arma::mat    invSigma22;
    arma::mat    Sig_11,Sig_12,Sig_21,Sig_22;
    arma::vec    Mu_1,Mu_2;
    double       det_22;

    arma::mat    Covariance_c;
    arma::colvec Mean_c;


};

}


#endif
