#include <statistics/distributions/gmm.h>
#include <boost/lexical_cast.hpp>
#include <statistics/distributions/distributions.h>

using namespace stats;

////////////////////////////
//		   GMM			  //
////////////////////////////

GMM::GMM(){
    K = 0;
    D = 0;
    name = "NONE";
}

GMM::GMM(std::string path_to_parameter_folder){
    this->name = name;
    load(path_to_parameter_folder);
}

GMM::GMM(const arma::colvec &weights, const std::vector<arma::vec>& Mu, const std::vector<arma::mat>& Sigma)
{

    pi          = weights;
    Means       = Mu;
    Covariances = Sigma;


}

void GMM::likelihood(const arma::mat& X,arma::vec& y){
    assert(X.n_rows ==  y.n_elem);

    for(std::size_t r = 0; r < X.n_rows;r++){
        for(std::size_t k = 0; k < K;k++){
            y(r) += pi(k) *  stats::mvnpdf(X.row(r),Means[k],Covariances[k]);
        }
    }

}

void GMM::expection(arma::colvec& x) const{
    x.zeros();
    for(std::size_t k = 0; k < K;k++){
        x += Means[k] * pi(k);
    }
}


void GMM::load(const std::string& file){


    arma::mat pi,mu;
    std::cout<< "=== loading Gaussian Mixture Model === " << std::endl;

    if(!pi.load(file + "/" + "priors.txt")){
        std::cout<< "***failed to load: " << file + "/" +  "priors.txt"  << std::endl;
    }else{
        std::cout<< "   loaded: " << file + "/" + "priors.txt" << std::endl;
    }

    if(!mu.load(file + "/" +  "mu.txt")){
        std::cout<< "***failed to load: " << file + "/" +  "mu.txt" << std::endl;
    }else{
        std::cout<< "   loaded: " << file + "/" +  "mu.txt" << std::endl;
    }

    K = pi.n_cols;
    D = mu.n_rows;

    std::string path_sigma = file + "/Sigmas/";
    std::string full_path_sigma;
    for(std::size_t i = 0; i < K; i++){
        full_path_sigma =  path_sigma + "sigma_" + boost::lexical_cast<std::string>(i+1) + ".txt";
        if(!Covariances[i].load(full_path_sigma)){
            std::cout<< "***failed to load: " << full_path_sigma<< std::endl;
        }else{
            //    std::cout<< "   loaded: " << "sigma_" + boost::lexical_cast<std::string>(i+1) + ".txt" << std::endl;
        }
    }


    std::cout<< "finished loading parameters" << std::endl;
    std::cout<< "K:       "  << K << std::endl;
    std::cout<< "D:       "  << D << std::endl;
    std::cout<< "priors:  (" << pi.n_rows << " x " << pi.n_cols << ")" << std::endl;
    std::cout<< "mu:      (" << mu.n_rows << " x " << mu.n_cols << ")" << std::endl;
    std::cout<< "sigma_0: (" << Covariances[0].n_rows << " x " << Covariances[0].n_cols << ")" << std::endl;

    // arma::vec weights(K);
    // std::vector<arma::vec> Mu(K);
    this->pi.resize(K);

    std::cout<< "#1" << std::endl;

    for(std::size_t k = 0; k < K; k++){
        this->pi(k) = pi(k);
        Means[k].resize(D);
        //  std::cout<< "weights("<<k<<"): " << weights(k) << std::endl;
    }

    // std::cout<< "#2" << std::endl;
    // std::cout<< "Mu.size(): " << Mu.size() << std::endl;

    arma::colvec tmp(D);
    for(std::size_t k = 0; k < K;k++){
        //    std::cout<< "mu("<<k<<")" << std::endl;
        for(std::size_t d = 0; d < D; d++){
            tmp(d) = mu(d,k);
            //      std::cout<< " " << mu(d,k) << std::endl;
        }
        //  std::cout<< "-" << std::endl;
        assert(tmp.is_finite());
        Means[k] = tmp;
    }

    std::cout<< "#3" << std::endl;


    std::cout<< "---> end loading covariance matrix" << std::endl;

    std::cout<< "=== Succesfully loaded Gaussian Mixture Model Parameters ===" << std::endl;


}

void GMM::sample(arma::mat& X){
    std::size_t k;
    arma::vec tmp;
    dist = boost::random::discrete_distribution<>(pi);
    for(std::size_t i = 0; i < X.n_rows;i++){

        k  =  dist(generator);
        A[k]*arma::randn<arma::vec>(D);
        tmp = Means[k] + A[k]*arma::randn<arma::vec>(D);
        X.row(i) = tmp.st();
    }
}


void cGMM::condition(const GMM& gmm_in,const std::vector<std::size_t>& in,const std::vector<std::size_t>& out){

    gaussian_c.resize(gmm_in.K);

    gmm_c.pi.resize(gmm_in.K);
    gmm_c.Means.resize(gmm_in.K);
    gmm_c.Covariances.resize(gmm_in.K);



    for(std::size_t k = 0; k < gmm_in.K;k++){
        // P( x_a | X_b)
        gaussian_c[k].condition(gmm_in.Means[k],gmm_in.Covariances[k],in,out);
        gmm_c.Covariances[k] = gaussian_c[k].Sigma_1c2;
    }

}


void cGMM::condition(const arma::colvec& x_in, const GMM &gmm_in){

    for(std::size_t k = 0; k < gmm_in.K;k++){
        // \mu_a + Sig_12 * Sig_22^{-1} * (x - \mu_b)
        gaussian_c[k].mu_condition(x_in);
        gmm_c.Means[k] = gaussian_c[k].Mean_c;
        gmm_c.pi(k)    = stats::mvnpdf(x_in,gaussian_c[k].Mu_2,gaussian_c[k].invSigma22,gaussian_c[k].det_22);
    }

    gmm_c.pi = (gmm_in.pi %  gmm_c.pi) / arma::sum( gmm_c.pi);
    gmm_c.pi =  gmm_c.pi / arma::sum(gmm_c.pi);

}
