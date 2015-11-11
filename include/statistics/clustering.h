/*
 * clustering.h
 *
 *  Created on: Jul 12, 2013
 *      Author: guillaume
 */

#ifndef CLUSTERING_H_
#define CLUSTERING_H_

#include <vector>
#include <iostream>
#include <ctime>
#include <mlpack/methods/kmeans/kmeans.hpp>
#include <statistics/distributions/distributions.h>
#include <statistics/distributions/gmm.h>

#include <statistics/metric/distance_metric.hpp>

#include <armadillo>

using namespace std;

class EM{

public:

	EM();

	void set(int N, int K);

	void fit(const arma::mat& data, std::vector<arma::vec>& means,std::vector<arma::mat>& covariances, arma::vec pi);

private:

	void compute_gamma(const arma::mat& data, std::vector<arma::vec>& means,std::vector<arma::mat>& covariances, arma::vec pi);

	void phi(const arma::mat& x,const arma::vec& mean, const arma::mat& cov,arma::vec& probabilities);

private:

	// responsability
	arma::mat gamma;
	arma::mat Nk;
    int N,K;



};

template<typename Type>
class Weighted_Kmeans{

public:

    Weighted_Kmeans(){
        K = 2;
    }

    void cluster(const arma::Mat<Type>& X,arma::colvec& weights){
         mlpack::metric::WSquaredEuclideanDistance WL2(weights);
        // mlpack::kmeans::KMeans<mlpack::metric::WSquaredEuclideanDistance> kw2(1000,1.0,WL2);
         mlpack::kmeans::KMeans<mlpack::metric::EuclideanDistance> kw(100,1.0);
        //mlpack::kmeans::KMeans<> kw;

         assignments.resize(X.n_cols);
         kw.Cluster(X,K,assignments,centroids,false,false);
         //kw.Cluster(X,K,assignments,centroids,false,false);
    }

public:

    std::size_t                K;
    arma::Col<std::size_t>     assignments;
    arma::Mat<Type>            centroids;

};



class Clustering{

public:

	Clustering();

    void kmeans(const arma::mat &data, const arma::vec &w=arma::vec());



    void compute_mixture_model(GMM& gmm, const arma::mat& data, const arma::vec& w);

    ///
    /// \brief mixture_model
    /// \param gmm   model for parameters to be fit to
    /// \param data (N x D)
    /// \param w    (N x 1) weights of each data point [0..1]
    ///
    void mixture_model(GMM& gmm,const arma::mat &data, const arma::vec &w);

    void setInitCentroids(arma::mat& centers);

    void LogLikelihood(const arma::mat& data, const arma::vec& w);

private:

    void one_mixture_component(GMM& gmm, const arma::mat &data);

    void multiple_components(GMM& gmm, const arma::mat& data);

    void phi(const arma::mat& x,const arma::vec& mean, const arma::mat& cov,arma::vec& probabilities);

    bool isIndefinite(const arma::vec& eigenvalues);

	void compute_covariance();


public:

    arma::Col<size_t>             assignments;
    std::vector<arma::mat >       covariances;
    std::vector<arma::vec>        means;
    arma::vec                     pi;
    arma::mat                     I;
    arma::mat                     centroids;
    mlpack::kmeans::KMeans<>      cluster_kmeans;
    arma::mat                     particle_hw;

private:
    int                           n_rows;
    int                           n_cols;
    std::size_t                   K;

    arma::vec                     N;
};





#endif /* CLUSTERING_H_ */
