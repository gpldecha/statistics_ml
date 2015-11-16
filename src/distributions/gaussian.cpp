#include <statistics/distributions/gaussian.h>
#include <statistics/distributions/distributions.h>

using namespace stats;

/// ========================  Gaussian function ==============================


Gaussian::Gaussian():gen(boost::variate_generator<ENG,DIST>(ENG(static_cast<unsigned int>(std::time(0))),DIST(0,1))),normal(boost::normal_distribution<double>(0,1))
{
    D = 0;
    gen.engine().seed(static_cast<unsigned int>(std::time(0)));
}

Gaussian::Gaussian(const arma::vec& Mean, const arma::mat& Cov):Mean(Mean),Covariance(Cov),gen(boost::variate_generator<ENG,DIST>(ENG(),DIST(0,1))){
    D = Mean.n_elem;
    if(D > 1){
        try{
            A = arma::chol(this->Covariance);
        }catch(std::runtime_error e){
            std::cout<< "chol did not work" << std::endl;
            this->Covariance.eye() * 0.00001;
            A = arma::chol(this->Covariance);
        }
        z.resize(Mean.n_elem);
        det = arma::det(Cov);
    }else{
        gen = boost::variate_generator<ENG,DIST>(ENG(),DIST((this->Mean)(0),(this->Covariance)(0,0)));

    }

    invCovariance = arma::inv(Covariance);


}

void Gaussian::fit(const arma::mat& data)
{
    if(data.size() > 0){
        D = data.n_rows;
        Mean = arma::mean(data);
        Covariance  = arma::cov(data);
        A    = arma::chol(Covariance);
        det  = arma::det(Covariance);
    }
}

double Gaussian::likelihood(const arma::colvec& x){
    if(Mean.n_elem > 1){
        //                                          (3 x 3) *  (3 x 1)
        return stats::mvnpdf(x,Mean,invCovariance,det);
    }else{
        double mu = Mean(0);
        double x_ = x(0);
        double var = Covariance(0,0);
        return 1.0/(sqrt(2*M_PI) * sqrt(var)) * exp(-0.5*(x_ - mu)*(x_ - mu)/sqrt(var));
    }
}

arma::vec Gaussian::sample(){
    for(unsigned int i = 0; i < Mean.n_rows;i++){
        z[i] = gen();
    }
    //  (3x1) + (3x3) * (3x1)
    return Mean + (A*z);
}


/// =========================== Conditional Gaussian function ==============================




void Gaussian_c::mu_condition(const arma::colvec& x){
   Mean_c   = Mu_1   + Sig_12 * invSigma22 * ( x - Mu_2 );
}

void Gaussian_c::condition(const arma::colvec& Mean, const arma::mat& Covariance, const std::vector<std::size_t>& in, const std::vector<std::size_t>& out){
    std::size_t outSize = out.size();
    std::size_t inSize  =  in.size();

    Sig_11.resize(outSize,outSize);
    Sig_12.resize(outSize,inSize);
    Sig_21.resize(inSize,outSize);
    Sig_22.resize(inSize,inSize);
    Mu_1.resize(outSize);
    Mu_2.resize(inSize);

    getBlock(Sig_11,Covariance,out,out);
    getBlock(Sig_12,Covariance,out,in);
    getBlock(Sig_21,Covariance,in,out);
    getBlock(Sig_22,Covariance,in,in);
    getBlock(Mu_1,Mean,out);
    getBlock(Mu_2,Mean,in);

    invSigma22 = arma::inv(Sig_22);
    det_22     = arma::det(invSigma22);
    Sigma_1c2  = Sig_11 - Sig_12 * invSigma22 * Sig_21;
}

void Gaussian_c::getBlock(arma::mat& A_xx,const arma::mat& A,const std::vector<std::size_t>& dim1,const std::vector<std::size_t>& dim2){
    for(std::size_t i = 0; i < dim1.size();i++){
        for(std::size_t j = 0; j < dim2.size();j++){
            A_xx(i,j) = A(dim1[i],dim2[j]);
        }
    }
}

void Gaussian_c::getBlock(arma::vec& Mu_xx,const arma::colvec& Mu, const std::vector<std::size_t> &dim){
    for(std::size_t i = 0; i < dim.size();i++){
        Mu_xx(i) = Mu(dim[i]);
    }
}
