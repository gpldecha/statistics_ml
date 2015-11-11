#include <statistics/clustering.h>
#include <armadillo>

int main(int argc, char** argv){


    std::cout<< "=== weighted K-means test === " << std::endl;

    Weighted_Kmeans<double> weighted_kmeans;


    arma::mat X = arma::randu<arma::mat>(3,100);
    arma::colvec weights(100,arma::fill::ones);

    weighted_kmeans.cluster(X,weights);


    weighted_kmeans.centroids.print("centroids");




    return 0;
}
