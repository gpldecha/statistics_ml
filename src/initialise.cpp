#include <statistics/initialise.h>
#include <limits>
#include <iostream>
#include <chrono>
#include <algorithm>
#include <stdlib.h>

Initialise::Initialise():gen(static_cast<unsigned int>(std::time(0)))
{
}


void Initialise::findInitialisation(const arma::mat &data, arma::mat& centers, const arma::colvec &w){

    std::cout<< "findInitialisation -in- " << std::endl;

    K = centers.n_rows;
    N = data.n_rows;
    D.resize(N);

   /* if(K > N){
        centers = data;
    }else{*/


        U = boost::random::uniform_int_distribution<>(0,N);

        index = U(gen);
        centers(0,0) = data(index,0);
        centers(0,1) = data(index,1);
        centers(0,2) = data(index,2);
        k_current=1;

        for(unsigned int k = 1; k < K; k++) {
            compute_distances(data,centers,w);
        }
    //}

    std::cout<< "findInitialisation -out- " << std::endl;

}


void Initialise::compute_distances(const arma::mat& data, arma::mat &centers, const arma::colvec& w){

    sum = 0;
    for(unsigned int i = 0; i < N; i++){
        D[i] = getDistToClosestCenter(centers,data.row(i).st());
        if(w.n_elem != 0){
            D[i] = D[i]*D[i]*w(i);
        }else{
           D[i] = D[i]*D[i];
        }
        sum = sum + D[i];
    }


    //double max_elem = (*std::max_element(D.begin(),D.end()));
   // std::cout<< "sample -1-" << std::endl;
    boost::random::discrete_distribution<> pdf(D.begin(),D.end());
    index = pdf(gen);
   // std::cout<< "index: " << index << std::endl;
    //index = sample_discrete(D,max_elem);
    //std::cout<< "sample -2- " << std::endl;
    centers.row(k_current) = data.row(index);
    k_current++;

}


inline double Initialise::getDistToClosestCenter(const arma::mat& centers, const arma::colvec& p){
    double closest = std::numeric_limits<double>::max();
    double d = 0;
    for(int k=0; k < k_current;k++){
        d = arma::norm(centers.row(k).st() - p,2);
        if(d < closest){
            closest = d;
        }
    }
    return closest;
}



