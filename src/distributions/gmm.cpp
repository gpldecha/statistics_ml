#include <statistics/distributions/gmm.h>
#include <boost/lexical_cast.hpp>

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

GMM::GMM(const arma::vec &weights,const std::vector<arma::vec>& Mu,const std::vector<arma::mat>& Sigma)
{
    setParam(weights,Mu,Sigma);
}

void GMM::likelihood(const arma::mat& X,arma::vec& y){
 /*   assert(X.n_rows ==  y.n_elem);
    for(std::size_t r = 0; r < X.n_rows;r++){

        for(std::size_t k = 0; k < K;k++){
            y(r) += Priors[k] * gaussians[k].P(X.row(r));
        }

    }*/


}

void GMM::P(const arma::mat& X,arma::vec& L){
    for(std::size_t i = 0; i < X.n_rows;i++){
        L(i) = gmm.Probability(X.row(i).st());
    }
}

double GMM::nlikelihood(const arma::vec& x){
    return -std::log(gmm.Probability(x));
}

double GMM::getPriors(std::size_t i){
    return (gmm.Weights())(i);
}

arma::vec& GMM::getMu(std::size_t i){
    return gmm.Means()[i];
}

arma::mat& GMM::getSigma(std::size_t i){
    return gmm.Covariances()[i];
}

void GMM::setMu(const arma::vec& mu, std::size_t k){
    assert(k < gmm.Means().size());
    gmm.Means()[k] = mu;
}


void GMM::condition(GMM& gmm_out, arma::vec& x, const std::vector<std::size_t> &in, const std::vector<std::size_t> &out){

    std::vector<double> w(K);

    double sum_P = 0;

    Gaussian tmp,tmp2;

    for(std::size_t k = 0; k < K;k++){

        gaussian.SetMean(gmm.Means()[k]);
        gaussian.SetCov(gmm.Covariances()[k]);

        tmp = gaussian.condition(x,in,out);

        //std::cout<< "-- 2" << std::endl;
        gmm_out.gmm.Means()[k]        = tmp.Mean;
        gmm_out.gmm.Covariances()[k]  = tmp.Cov;

        tmp2.SetMean(tmp.Mu_2);
        tmp2.SetCov(tmp.Sig_22);
        //std::cout<< "-- 3" << std::endl;

       // x.print("x");
       // tmp.Mean.print("Mean");
       // tmp.Cov.print("Cov");

       // exit(0);

        w[k] = tmp2.P(x);

       // std::cout<< "w[" << k << "]: " << w[k] << std::endl;


       // std::cout<< "--3.5"<< std::endl;

        sum_P += w[k];
       // std::cout<< "-- 4" << std::endl;


    }

   // std::cout<< "sum_P: " << sum_P << std::endl;

   // exit(0);

    // std::cout<< "-- 5" << std::endl;

    double sum_pi = 0;
    for(std::size_t k = 0; k < K;k++){
        gmm_out.gmm.Weights()[k] = gmm.Weights()[k] * w[k] / sum_P;
        sum_pi +=  gmm_out.gmm.Weights()[k];
    }
    for(std::size_t k = 0; k < K; k++){
       gmm_out.gmm.Weights()[k] = gmm_out.gmm.Weights()[k]/sum_pi;
    }
    // std::cout<< "-- 6" << std::endl;

}

void GMM::expection(arma::vec& x){
    x.zeros();
    for(std::size_t k = 0; k < K;k++){
        x += gmm.Means()[k] * gmm.Weights()[k];
    }
}


/*
arma::vec &GMM::E(){
    expectation.zeros();
    for(unsigned int k = 0; k < K;k++){
        expectation += gaussians[k].Mean * Priors[k];
    }
    return expectation;
}
*/

void GMM::setParam(const arma::vec &weights,const std::vector<arma::vec>& Mu,const std::vector<arma::mat>& Sigma){
    std::cout<< "set-parameters" << std::endl;

    gmm = mlpack::gmm::GMM<double>(Mu,Sigma,weights);

    std::cout<< "-1-" << std::endl;
    K   = weights.n_elem;
    std::cout<< "-2-" << std::endl;
    D   = Mu[0].n_elem;
    std::cout<< "-3-" << std::endl;
    A.resize(weights.n_elem);
    std::cout<< "-4-" << std::endl;
    for(std::size_t i = 0; i < weights.n_elem;i++){
        A[i] = arma::chol(Sigma[i]);
    }
    std::cout<< "-5-" << std::endl;
}

void GMM::print(){

    gmm.Weights().print("weights");

    /*for(std::size_t i = 0; i < gmm.Means().size();i++){
        gmm.Means()[i].print("mean(" + boost::lexical_cast<std::string>(i) + ")");
    }*/

 /*   for(std::size_t i = 0; i < gmm.Covariances().size();i++){
        gmm.Covariances()[i].print("covariance(" + boost::lexical_cast<std::string>(i) + ")");

    }*/


}

/*
arma::vec& GMM::getMu(unsigned int k){
    return gaussians[k].Mean;
}

arma::mat& GMM::getSigma(unsigned int k){
    return gaussians[k].Cov;
}

std::vector<double>& GMM::getPriors(){
    return Priors;
}

double GMM::getPriors(unsigned int k){
    return Priors[k];
}*/

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
    std::vector<arma::mat> Sigmas(K);
    for(std::size_t i = 0; i < K; i++){
        full_path_sigma =  path_sigma + "sigma_" + boost::lexical_cast<std::string>(i+1) + ".txt";
        if(!Sigmas[i].load(full_path_sigma)){
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
    std::cout<< "sigma_0: (" << Sigmas[0].n_rows << " x " << Sigmas[0].n_cols << ")" << std::endl;

    arma::vec weights(K);
    std::vector<arma::vec> Mu(K);

    std::cout<< "#1" << std::endl;

    for(std::size_t k = 0; k < K; k++){
       weights(k) = pi(k);
       Mu[k].resize(D);
     //  std::cout<< "weights("<<k<<"): " << weights(k) << std::endl;
    }

    weights.print("weights");

   // std::cout<< "#2" << std::endl;
   // std::cout<< "Mu.size(): " << Mu.size() << std::endl;

    arma::vec tmp(D);
    for(std::size_t k = 0; k < K;k++){
     //    std::cout<< "mu("<<k<<")" << std::endl;
        for(std::size_t d = 0; d < D; d++){
            tmp(d) = mu(d,k);
      //      std::cout<< " " << mu(d,k) << std::endl;
        }
      //  std::cout<< "-" << std::endl;
        assert(tmp.is_finite());
        Mu[k] = tmp;
    }

        std::cout<< "#3" << std::endl;


    std::cout<< "---> end loading covariance matrix" << std::endl;
    std::cout<< "weights:  (" << weights.n_rows << " x " << weights.n_cols << ")" << std::endl;
    setParam(weights,Mu,Sigmas);


    std::cout<< "=== Succesfully loaded Gaussian Mixture Model Parameters ===" << std::endl;


}

void GMM::set_name(const std::string& name){
    this->name = name;
}

void GMM::clear(){


}


void GMM::sample(arma::mat& X){
    std::size_t k;
    arma::vec tmp;
    dist = boost::random::discrete_distribution<>(gmm.Weights());
    for(std::size_t i = 0; i < X.n_rows;i++){

        k  =  dist(generator);
        A[k]*arma::randn<arma::vec>(D);
        tmp = gmm.Means()[k] + A[k]*arma::randn<arma::vec>(D);
        X.row(i) = tmp.st();
    }
}

