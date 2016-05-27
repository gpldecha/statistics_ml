/*
 * clustering.cpp
 *
 *  Created on: Jul 12, 2013
 *      Author: guillaume
 */


#include <statistics/clustering.h>
#include <algorithm>
#include <limits>

using namespace stats;

Clustering::Clustering(){

    cluster_kmeans = mlpack::kmeans::KMeans<>(300);

   // mlpack::kmeans::KMeans<mlpack::metri>      cluster_kmeans2();
   // mlpack::metric::EuclideanDistance
}


// centers is D x K
void Clustering::setInitCentroids(arma::mat& centers){
	centroids = centers;
	K = centroids.n_cols;

    I = I.eye(3,3);
    I = I * 0.0001;


}

void Clustering::kmeans(const arma::mat &data, const arma::vec& w){
   // arma_data = data.st();
    assignments.resize(data.n_cols);

   // std::cout<< "in cluster" << std::endl;
   // std::cout<< "data:      (" << data.n_rows << " x " << data.n_cols << ") " << std::endl;
   // std::cout<< "centroids: (" << centroids.n_rows << " x " << centroids.n_cols << ") " << std::endl;
   // std::cout<< "assignmen: (" << assignments.n_elem << ") " << std::endl;
  //  cluster_kmeans.Cluster(data,K,assignments,centroids,false,true);
  //  std::cout<< "cluster finish" << std::endl;
}

void Clustering::compute_mixture_model(GMM& gmm, const arma::mat& data, const arma::vec& w){
    covariances.resize(K);
    means.resize(K);
    pi.resize(K);

    std::size_t k;
    // set to zero
    for(k = 0; k < K;k++){
        means[k].zeros(3);
        covariances[k].zeros(3,3);
    }

    for(std::size_t r = 0; r < data.n_rows;r++){
        k           = assignments(r);
        means[k]    = means[k] + data.row(r).st();
        N(k)        = N(k) + 1;
        pi[k]       = pi[k] + w(r);
    }

}

void Clustering::mixture_model(GMM &gmm, const arma::mat &data, const arma::vec &w){

    if(K == 1){
        one_mixture_component(gmm,data);
    }else{
        multiple_components(gmm,data);
    }

  /*  std::size_t k;
    arma::mat   x;
    // set to zero
    for(k = 0; k < K;k++){
        means[k].zeros(3);
        covariances[k].zeros(3,3);
        pi[k]   = std::numeric_limits<double>::min();
    }

    for(std::size_t r = 0; r < data.n_rows;r++){
        k           = assignments(r);
        means[k]    = means[k] + data.row(r).st();
        N(k)        = N(k) + 1;
        pi[k]       = pi[k] + w(r);
    }

    double sum_pi=0;
    for(k = 0; k < K;k++){
        means[k]    = means[k]/N(k);
        pi[k]       = pi[k]/sum_w;
        sum_pi      = sum_pi + pi[k];
    }

    for(std::size_t r = 0; r < data.n_rows;r++){
        k               = assignments(r);
        x               = data.row(r).st();
        covariances[k]  = covariances[k] + (x - means[k]) * (x - means[k]).st() ;
	}

    for(k = 0; k < K;k++){
        covariances[k] = (covariances[k])/N(k);
        covariances[k] = covariances[k] + I;
        covariances[k] = 0.5*(covariances[k] + covariances[k].st());
        pi[k]          = pi[k]/sum_pi;
    }*/



}

void Clustering::one_mixture_component(GMM& gmm,const arma::mat &data){

    covariances.resize(1);
    means.resize(1);
    pi.resize(1);
    means[0]        = arma::mean(data,0).st();
    pi(0)           = 1;
    covariances[0].eye(3,3);

   // std::cout<< "compute covariance" << std::endl;
   // data.print("data");
    arma::colvec x;
    for(std::size_t r = 0; r < data.n_rows;r++){
        x               = data.row(r).st();

        covariances[0]  = covariances[0] + (x - means[0]) * (x - means[0]).st() ;
    }

    covariances[0] = covariances[0]/((double)data.n_rows);
  //  covariances[0].print("covariances[0]");

    covariances[0] = covariances[0] + I;
    covariances[0] = 0.5*(covariances[0] + covariances[0].st());

    //covariances[0].print("covariances[0]");

 //   gmm.setParam(pi,means,covariances);

   // gmm.print();
}

void Clustering::multiple_components(GMM& gmm, const arma::mat& data){

   /* data.print("data");
    std::cout<< "K: " << K << std::endl;
    std::cout<< "centroids: " << centroids.n_rows << " x " << centroids.n_cols << std::endl;
*/
    covariances.resize(K);
    means.resize(K);
    pi.resize(K);

    arma::vec x;
    std::size_t k;
    // set to zero
    for(k = 0; k < K;k++){
        means[k]  = centroids.col(k);
        covariances[k].zeros(3,3);
        pi(k)   = 0;//std::numeric_limits<double>::min();
    }

    for(std::size_t r = 0; r < data.n_rows;r++){
        k               = assignments(r);
        x               = data.row(r).st();
        covariances[k]  = covariances[k] + (x - means[k]) * (x - means[k]).st() ;
        pi(k)           = pi(k) + 1;
    }


    for(k = 0; k < K;k++){
        covariances[k] = (covariances[k])/pi(k);
        covariances[k] = covariances[k] + I;
        covariances[k] = 0.5*(covariances[k] + covariances[k].st());
        pi(k)          = pi(k) + std::numeric_limits<double>::min();
    }



    pi = pi / arma::sum(pi);
   /* pi.print("pi");
    for(std::size_t i = 0; i < covariances.size();i++){
        covariances[i].print("cov(" + boost::lexical_cast<std::string>(i) + ")");
    }
*/

    //gmm.setParam(pi,means,covariances);


}

/*
 void Clustering::setPriorsGMM(const arma::mat& data,const arma::vec& w){

     pi.resize(gmm.K);

      for(unsigned int k = 0; k < gmm.K;k++){
          pi[k] = 0;
      }

    for(unsigned int i = 0; i < data.n_rows;i++){
        for(unsigned int k = 0; k < gmm.K;k++){
            pi[k] = pi[k] +  mvnpdf(data.row(i).st(),means[k],covariances[k]) * w(i);
        }
     }
     double sum_k = 0;
     for(unsigned int k = 0; k < gmm.K;k++){
         sum_k = sum_k + pi[k];
     }
     for(unsigned int k = 0; k < gmm.K;k++){
         pi[k]=pi[k]/sum_k;
     }

     //gmm.setPriors(pi);

     if(pi.size()==1){
        mostLikelyModeIndex = 0;
     }else{
            max_value = 0;
            for(unsigned int i = 0; i < gmm.K;i++){
                if(pi[i] > max_value){
                    mostLikelyModeIndex = i;
                    max_value = pi[i];
                }
            }


     }

 }

*/
bool Clustering::isIndefinite(const arma::vec& eigenvalues){
    for(std::size_t i = 0; i < eigenvalues.n_elem;i++){
        if(eigenvalues(i) < 0){
            return true;
        }
    }

    return false;
}


void Clustering::LogLikelihood(const arma::mat& data, const arma::vec& w)
{
  size_t gaussians = means.size();
  arma::vec phis;
  arma::mat likelihoods(gaussians, data.n_cols);
  double sum = 0;
  for (size_t i = 0; i < gaussians; i++)
  {
    //mlpack::gmm::phi(data, means[i], covariances[i], phis);
    //likelihoods.row(i) = w(i) * trans(phis);
    //pi[i] = arma::sum(likelihoods.row(i));
   // sum = sum + pi[i];
  }

  for (size_t i = 0; i < gaussians; i++)
  {
      pi[i] = pi[i]/sum;
  }

}

void Clustering::phi(const arma::mat& x,const arma::vec& mean, const arma::mat& cov,arma::vec& probabilities)
{
      // ( 1 x D)
     arma::vec mu = mean;
      //     (N x D)    (N x D) - ( 1 X D repmat N)
     arma::mat diffs = (x - (mean.st() * arma::ones<arma::rowvec>(x.n_rows))).st();

     std::cout<< "here" << std::endl;
      // Now, we only want to calculate the diagonal elements of (diffs' * cov^-1 *
      // diffs).  We just don't need any of the other elements.  We can calculate
      // the right hand part of the equation (instead of the left side) so that
      // later we are referencing columns, not rows -- that is faster.
      arma::mat rhs = -0.5 * inv(cov) * diffs;
      arma::vec exponents(diffs.n_cols); // We will now fill this.
      for (size_t i = 0; i < diffs.n_cols; i++)
        exponents(i) = exp(accu(diffs.unsafe_col(i) % rhs.unsafe_col(i)));

      probabilities = pow(2 * M_PI, (double) mean.n_elem / -2.0) *
          pow(det(cov), -0.5) * exponents;
}



EM::EM(){

}

void EM::set(int _N, int _K){
	N = _N;
	K = _K;
	gamma.reshape(N,K);

}

void EM::phi(const arma::mat& x,const arma::vec& mean, const arma::mat& cov,arma::vec& probabilities)
{
	 arma::mat diffs = x - (mean * arma::ones<arma::rowvec>(x.n_cols));

	  // Now, we only want to calculate the diagonal elements of (diffs' * cov^-1 *
	  // diffs).  We just don't need any of the other elements.  We can calculate
	  // the right hand part of the equation (instead of the left side) so that
	  // later we are referencing columns, not rows -- that is faster.
	  arma::mat rhs = -0.5 * inv(cov) * diffs;
	  arma::vec exponents(diffs.n_cols); // We will now fill this.
	  for (size_t i = 0; i < diffs.n_cols; i++)
	    exponents(i) = exp(accu(diffs.unsafe_col(i) % rhs.unsafe_col(i)));

	  probabilities = pow(2 * M_PI, (double) mean.n_elem / -2.0) *
	      pow(det(cov), -0.5) * exponents;
}


void EM::compute_gamma(const arma::mat& data, std::vector<arma::vec>& means,std::vector<arma::mat>& covariances, arma::vec pi){
	arma::vec phis;
	std::cout<< "data: " << data.n_rows << "x" << data.n_cols << std::endl;
	for(int i = 0; i < K;i++){
		arma::vec mu = means[i];
		arma::mat cov =  covariances[i];
		phi(data, mu, cov, phis);
		gamma.col(i) = phis;
	}

	arma::vec sum =  arma::sum(gamma,1);
	for(int i = 0; i < N; i++){
		for(int c = 0; c < K; c++){
			gamma(i,c) =  gamma(i,c) / sum(i);
		}
	}

	gamma.row(0).print("row 0");
	std::cout<< "sum row 0: " << arma::sum(gamma.row(0)) << std::endl;


	Nk = arma::sum(gamma,0);

}

// data is D x N
void EM::fit(const arma::mat& data, std::vector<arma::vec>& means,std::vector<arma::mat>& covariances, arma::vec pi){

		// E-step
		compute_gamma(data,means,covariances,pi);

		// M-step

		// means

        for(unsigned int i = 0; i < data.n_cols; i++){
			for(int c =0; c < K; c++){
				means[c] = gamma(i,c) * data.col(i);
			}
		}

		for(int c =0; c < K; c++){
			means[c] = (1/Nk(c)) * means[c];
			covariances[c].zeros();
		}



		for(unsigned int i = 0; i < data.n_cols;i++){
			for(int c =0; c < K; c++){
				covariances[c] = covariances[c] + gamma(i,c) * ((data.col(i) - means[c]) * (data.col(i) - means[c]).st()) ;
			}
		}

		for(int c =0; c < K; c++){
			covariances[c] = (1/Nk(c)) * covariances[c];
			pi(c) = Nk(c)/(double)N;
		}

		covariances[0].print("cov 0");


}



