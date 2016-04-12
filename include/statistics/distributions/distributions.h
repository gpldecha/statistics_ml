/*
 * distributions.h
 *
 *  Created on: Nov 16, 2012
 *      Author: guillaume
 */

#ifndef DISTRIBUTIONS_H_
#define DISTRIBUTIONS_H_

// STL

#include <vector>


// Boost
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/normal_distribution.hpp>
#include <boost/random/uniform_real.hpp>
#include <boost/random/variate_generator.hpp>
#include <boost/random/discrete_distribution.hpp>
#include <boost/range.hpp>
#include <boost/array.hpp>

#include <armadillo>

namespace stats{

inline double mvnpdf(const arma::colvec& x,const arma::colvec& mu, const arma::mat& cov){
    return 1.0/(pow(2*M_PI,(double)cov.n_rows/2)*sqrt(arma::det(cov))) * exp(-0.5*arma::dot((x - mu),arma::inv(cov) * (x - mu)));
}

inline double mvnpdf(const arma::colvec &x, const arma::colvec &mu, const arma::mat& invcov,double det){
    return 1.0/(pow(2*M_PI,(double)mu.n_elem/2)*sqrt(det)) * exp(-0.5*arma::dot((x - mu), invcov * (x - mu)));
}


inline void regularise_covariance(arma::mat& cov,double min_x,double min_y,double min_z){
    if(cov(0,0) < min_x)
        cov(0.0) = min_x;
    if(cov(1,1) < min_y)
        cov(1,1) = min_y;
    if(cov(2,2) == min_z)
        cov(2,2) = min_z;
}

inline double random_d(double a, double b) {
    return a + (((double) rand()) / (double) RAND_MAX) * (b - a);
}

inline std::size_t random_i(std::size_t a, std::size_t b){
    return (b - a +1)*((double)rand()/(double)RAND_MAX) + a;
}

inline int sample_discrete(const std::vector<double> &p,double max){
    unsigned int size = p.size();
    unsigned int index = random_i(0,size);
    double beta = 0;
    beta = beta + random_d(0,2*max);
    while(beta > p[index]){
        beta = beta - p[index];
        index = (index + 1) % size;
    }
    if(index >= size){
        index = size-1;
    }

    return index;
}


class Distribution{

public:

    virtual arma::vec sample() = 0;

};

class Uniform : public Distribution {

public:
    Uniform();

    Uniform(const arma::vec3& origin,const arma::mat& orient,double length, double width, double height);

    void SetOrigin(const arma::vec& origin);

    void Set(double length,double width,double height);

    arma::vec GetOrigin();

    double getLength();

    double getWidth();

    double getHeight();

    virtual arma::vec sample();

    void draw();

    void translate(const arma::vec& T);


private:

    void init();

private:

    arma::vec origin;
    arma::mat orient;
    double length, width, height;
    boost::mt19937 generator;
    boost::uniform_real<double> uniform_x, uniform_y, uniform_z;


};

class MixUniform : public Distribution{

public:

    MixUniform();

    ~MixUniform();

    MixUniform(const std::vector<Uniform>& uniformList,const std::vector<double>& P);

    virtual arma::vec sample();

    void translate(const arma::vec& T);

private:

    std::vector<Uniform> uniformList;
    boost::mt19937 generator;
    boost::random::discrete_distribution<> dist;

};

class Epanechnikov : public Distribution{
public:
    Epanechnikov(int nx,double cx, int Ns);

    void SetMean(const arma::vec& Mean);

    void SetCov(const arma::mat& Cov);

    inline double K(double u);

    virtual arma::vec sample();
private:

    typedef boost::mt19937 ENG;
    typedef boost::normal_distribution<double> DIST;
    boost::variate_generator<ENG,DIST>   gen;
    arma::vec x_mean, x;
    arma::mat D;
    arma::vec e;
    int nx, Ns;
    double cx;
    double A;
    double h;
};


class Distance{

public:

    static inline double Hellinger(const arma::colvec& P, const arma::colvec& Q){
        return (1.0/sqrt(2.0)) * sqrt(arma::sum(arma::square(arma::sqrt(P) - arma::sqrt(Q))));
    }


    static inline double KL(const std::vector<double>& P, const std::vector<double>& Q){
        double kl = 0;
        const double epsilon = 0.0000000001;
        double p,q;
        for(unsigned int i = 0; i < P.size(); i++){
            p = std::max(Q[i],epsilon);
            q = std::max(P[i],epsilon);
            kl += (log(p) - log(q)) * p;
        }

        return kl;
    }

    static inline double KL(const arma::vec& P, const arma::vec& Q){
        double kl =0 ;
        double epsilon = 0.000000001;
        double p,q;
        for(unsigned int i = 0; i < P.n_elem;i++){
            p = std::max(P(i),epsilon);
            q = std::max(Q(i),epsilon);
            kl +=  (log(p) - log(q)) * p;
        }
        return kl;
    }

    static inline double JSD(const std::vector<double>& P,const std::vector<double> &Q){
        std::vector<double> M(P.size());
        assert(P.size() == Q.size());
        for(unsigned int i = 0; i < P.size();i++){
            M[i] = (P[i] + Q[i])/2;

        }
        return 0.5*KL(P,M) + 0.5*KL(Q,M);
    }

    static inline double JSD(const arma::vec& P, const arma::vec& Q){
        assert(P.n_elem == Q.n_elem);
        arma::vec M = (P + Q)/2;
        return 0.5 * KL(P,M) + 0.5 * KL(Q,M);
    }



};

}

#endif /* DISTRIBUTIONS_H_ */
