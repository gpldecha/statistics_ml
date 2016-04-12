#include <statistics/distributions/gmm.h>
#include <boost/lexical_cast.hpp>
#include <statistics/distributions/distributions.h>

using namespace stats;


void Load_param::load_scale(const std::string path_param){

    tmp.clear();
    if(!tmp.load(path_param + "/" + "scale.txt")){
        std::cout<< "   loaded: " << path_param + "/" +  "scale.txt" << std::endl;
        scale_.bscale = false;
    }else{
        std::cout<< "   loaded: " << path_param + "/" +  "scale.txt" << std::endl;
       // gmm.scale.h(1),gmm.scale.h(2),gmm.scale.h_min,gmm.scale.h_max
        scale_.bscale      = true;
        scale_.dim         = tmp(0);
        scale_.target_min  = tmp(1);
        scale_.target_max  = tmp(2);
        scale_.min_d       = tmp(3);
        scale_.max_d       = tmp(4);
    }
}


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


    arma::mat pi,mu,out_,in_;
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

    if(!out_.load(file + "/" + "out.txt")){
        std::cout<< "***failed to load: " << file + "/" +  "out.txt" << std::endl;
    }else{
        std::cout<< "   loaded: " << file + "/" +  "out.txt" << std::endl;
        out.resize(out_.n_elem);
        for(std::size_t i = 0; i < out_.n_elem;i++){
            out[i] = out_(i);
        }
    }

    if(!in_.load(file + "/" + "in.txt")){
        std::cout<< "   loaded: " << file + "/" +  "in.txt" << std::endl;
    }else{
        std::cout<< "   loaded: " << file + "/" +  "in.txt" << std::endl;
        in.resize(in_.n_elem);
        for(std::size_t i = 0; i < in_.n_elem;i++){
            in[i] = in_(i);
        }
    }




    K = pi.n_cols;
    D = mu.n_rows;

    std::cout<< "start loading sigmas" << std::endl;
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

    assert(Means.size() > 0);
    Means[0].print("means (0)");

   // std::cout<< "#3" << std::endl;

    //


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

  //  std::cout<< "cGMM::condition independent x" << std::endl;

    std::size_t D_c = in.size();
    std::size_t K   = gmm_in.K;

    arma::colvec           weights(K);
    std::vector<arma::vec> Means(K);
    std::vector<arma::mat> Covariances(K);


  //  std::cout<< "K: " << K << std::endl;
    gaussian_c.resize(K);

   // gmm_in.get_means()[0].print("gmm_in.get_means()[0]");
   // gmm_in.get_covariances()[0].print("gmm_in.get_covariances()[0]");


    for(std::size_t k = 0; k < gmm_in.K;k++){
        // P( x_a | X_b)
        gaussian_c[k].condition(gmm_in.get_means()[k],gmm_in.get_covariances()[k],in,out);
     //   std::cout<< "after conidtion["<<0<<"]" << std::endl;

        Means[k].resize(D_c);
        Covariances[k] = gaussian_c[k].Sigma_1c2;
       // std::cout<< "after Covariances["<<k<<"]" << std::endl;

    }

    gmm_c = GMM(weights,Means,Covariances);

}


void cGMM::condition(const arma::colvec& x_in, const GMM &gmm_in){

//    std::cout<< "cGMM::condition dependent x" << std::endl;

   // x_in.print("x_in");

    for(std::size_t k = 0; k < gmm_in.K;k++){
        // \mu_a + Sig_12 * Sig_22^{-1} * (x - \mu_b)
        gaussian_c[k].mu_condition(x_in);
        gmm_c.set_mu(k,gaussian_c[k].Mean_c);
        gmm_c.set_prior(k,gmm_in.get_weigts()[k] * stats::mvnpdf(x_in,gaussian_c[k].Mu_2,gaussian_c[k].invSigma22,gaussian_c[k].det_22));
    }


    gmm_c.set_prior( gmm_c.get_weigts() / arma::sum( gmm_c.get_weigts() + std::numeric_limits<double>::min()) );
}

void cGMM::print(const std::string &p) const{
    std::cout<< "=== cGMM " << gmm_c.name << "===" << std::endl;
    if(p == ""){
    gmm_c.get_weigts().print("pi");
    std::cout<< "K: " << gmm_c.K << std::endl;
    std::cout<< "D: " << gmm_c.D << std::endl;
    }else if(p == "in"){
        std::cout<< "in["<<gmm_c.in.size() << "]: ";
        for(std::size_t i = 0; i < gmm_c.in.size();i++){
            std::cout<< gmm_c.in[i] << " ";
        }
        std::cout<<std::endl;
    }else if(p == "out"){
        std::cout<< "out["<<gmm_c.out.size() << "]: ";
        for(std::size_t i = 0; i < gmm_c.out.size();i++){
            std::cout<< gmm_c.out[i] << " ";
        }
        std::cout<<std::endl;
    }

}
