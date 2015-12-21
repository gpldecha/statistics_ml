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

    assert(Mu.size() >= 1);

    K = pi.n_elem;
    D = Mu[0].n_elem;

}

const arma::colvec &GMM::get_weigts() const{
    return pi;
}

const std::vector<arma::colvec>& GMM::get_means() const{
    return Means;
}

const std::vector<arma::mat>& GMM::get_covariances() const{
    return Covariances;
}


void GMM::set_prior(const arma::colvec& weights){
    assert(weights.n_elem == K);
    pi=weights;

    if(pi.has_nan()){
        std::cout<< " weights have nan values, weights set to zero [GMM::set_prior]" << std::endl;
        pi.zeros();
    }

}

void GMM::set_prior(const std::size_t i,const double value){
    assert(i < K);
    pi(i) = value;
}

void GMM::set_mu(const std::size_t i,const arma::vec& mu){
    assert(i < K);
    Means[i] = mu;
}

void GMM::set_covariance(const std::size_t i,const arma::mat& covariance){
    assert(i < K);
    Covariances[i] = covariance;
}


std::vector<arma::colvec>& get_means();

std::vector<arma::mat>&    get_covariances();


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

    std::cout<< "start loading sigmas" << std::endl;
    mu.print("mu");

    Means.resize(K);
    Covariances.resize(K);


    std::string path_sigma = file + "/Sigmas/";
    std::string full_path_sigma;
    for(std::size_t i = 0; i < K; i++){
        full_path_sigma =  path_sigma + "sigma_" + boost::lexical_cast<std::string>(i+1) + ".txt";
        if(!Covariances[i].load(full_path_sigma)){
            std::cout<< "***failed to load: " << full_path_sigma<< std::endl;
        }else{
               std::cout<< "   loaded: " << "sigma_" + boost::lexical_cast<std::string>(i+1) + ".txt" << std::endl;
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

   // std::cout<< "#1" << std::endl;

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

   // std::cout<< "#3" << std::endl;


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

void GMM::print() const{
    std::cout<< "=== GMM " << name << "===" << std::endl;
    pi.print("pi");
    std::cout<< "K: " << K << std::endl;
    std::cout<< "D: " << D << std::endl;
}


void cGMM::condition(const GMM& gmm_in,const std::vector<std::size_t>& in,const std::vector<std::size_t>& out){

    std::cout<< "cGMM::condition independent x" << std::endl;

    std::size_t D_c = in.size();
    std::size_t K   = gmm_in.K;

    arma::colvec           weights(K);
    std::vector<arma::vec> Means(K);
    std::vector<arma::mat> Covariances(K);

    gaussian_c.resize(K);


    for(std::size_t k = 0; k < gmm_in.K;k++){
        // P( x_a | X_b)
        gaussian_c[k].condition(gmm_in.get_means()[k],gmm_in.get_covariances()[k],in,out);
        Means[k].resize(D_c);
        Covariances[k] = gaussian_c[k].Sigma_1c2;

    }

    gmm_c = GMM(weights,Means,Covariances);

}


void cGMM::condition(const arma::colvec& x_in, const GMM &gmm_in){

    std::cout<< "cGMM::condition dependent x" << std::endl;


    for(std::size_t k = 0; k < gmm_in.K;k++){
        // \mu_a + Sig_12 * Sig_22^{-1} * (x - \mu_b)
        gaussian_c[k].mu_condition(x_in);
        gmm_c.set_mu(k,gaussian_c[k].Mean_c);

       /* if(k == 0){
            gaussian_c[k].Mu_2.print("Mu_2");
            gaussian_c[k].invSigma22.print("invSigma22");
            std::cout<< "det_22: "<< gaussian_c[k].det_22 << std::endl;
            std::cout<< "gaussPDF: "  << stats::mvnpdf(x_in,gaussian_c[k].Mu_2,gaussian_c[k].invSigma22,gaussian_c[k].det_22) << std::endl;
        }*/
        gmm_c.set_prior(k,gmm_in.get_weigts()[k] * stats::mvnpdf(x_in,gaussian_c[k].Mu_2,gaussian_c[k].invSigma22,gaussian_c[k].det_22));

    }


    gmm_c.set_prior( gmm_c.get_weigts() / arma::sum( gmm_c.get_weigts() + std::numeric_limits<double>::min()) );

  /*  print();


    std::cout<< "pi[" << 0 << "]" <<gmm_c.get_weigts()[0] << std::endl;
    gmm_c.get_means()[0].print("means[0]");
    gmm_c.get_covariances()[0].print("covariance[0]");*/

}

void cGMM::print() const{
    std::cout<< "=== cGMM " << gmm_c.name << "===" << std::endl;
    gmm_c.get_weigts().print("pi");
    std::cout<< "K: " << gmm_c.K << std::endl;
    std::cout<< "D: " << gmm_c.D << std::endl;

}
