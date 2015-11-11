#ifndef INITIALISE_H_
#define INITIALISE_H_

#include <armadillo>
#include <random>
#include <vector>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_int_distribution.hpp>
#include <boost/random/uniform_real.hpp>
#include <boost/random/discrete_distribution.hpp>
#include <statistics/distributions/distributions.h>
#include <time.h>
#include <array>

class Initialise{


public:

    Initialise();

    void findInitialisation(const arma::mat& data, arma::mat &centers, const arma::colvec& w = arma::colvec());

private:

    void compute_distances(const arma::mat& data, arma::mat& centers, const arma::colvec &w);

    inline double getDistToClosestCenter(const arma::mat& centers, const arma::colvec &p);

private:

  boost::mt19937 gen;
  boost::uniform_real<double> U_beta;
  boost::random::uniform_int_distribution<> U;
  std::vector<double> D;
  std::size_t N,K,index;
  int k_current;
  double sum;


};



#endif
