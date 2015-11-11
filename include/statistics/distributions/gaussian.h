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

class Gaussian{

    typedef boost::mt19937 ENG;
    typedef boost::normal_distribution<double> DIST;

public:

    Gaussian();

    Gaussian(const arma::vec& Mean, const arma::mat& Cov);

    Gaussian condition(arma::vec& x,const std::vector<std::size_t>& in,const std::vector<std::size_t>& out);

    double P(const arma::vec& X);

    void fit(const arma::mat& data);

    void SetMean(const arma::vec &Mean);
    void SetCov(const arma::mat &Cov);

    arma::vec sample();

protected:

    void getBlock(arma::mat& A_xx,arma::mat& A,const std::vector<std::size_t>& dim1,const std::vector<std::size_t>& dim2);

    void getBlock(arma::vec &Mu_xx,arma::vec Mu,const std::vector<std::size_t>& dim);

public:

    arma::vec Mean;
    arma::mat Cov;
    double det;

private:

    double k;
    std::size_t D;
    arma::mat   A; //Cholesky of Cov
    arma::vec   z;
    boost::mt19937 generator;
    static boost::mt19937 gens;
    boost::variate_generator<ENG,DIST>   gen;
    boost::normal_distribution<double> normal;

public:

    arma::mat Sig_11,Sig_12,Sig_21,Sig_22;
    arma::vec Mu_1,Mu_2;

};


#endif
