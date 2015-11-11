#include <statistics/distributions/gaussian.h>

///////////////////////////
///    	 Gaussian	     //
///////////////////////////


Gaussian::Gaussian():gen(boost::variate_generator<ENG,DIST>(ENG(static_cast<unsigned int>(std::time(0))),DIST(0,1))),normal(boost::normal_distribution<double>(0,1))
{
    D = 0;
    gen.engine().seed(static_cast<unsigned int>(std::time(0)));
}

Gaussian Gaussian::condition(arma::vec &x,const std::vector<std::size_t>& in,const std::vector<std::size_t>& out){


    std::size_t outSize = out.size();
    std::size_t inSize = in.size();


    //std::cout<< "inSize : " << inSize << std::endl;
   // std::cout<< "outSize: " << outSize << std::endl;


    Sig_11.resize(outSize,outSize);
    Sig_12.resize(outSize,inSize);
    Sig_21.resize(inSize,outSize);
    Sig_22.resize(inSize,inSize);   // in
    Mu_1.resize(outSize);
    Mu_2.resize(inSize);

    getBlock(Sig_11,Cov,out,out);
    getBlock(Sig_12,Cov,out,in);
    getBlock(Sig_21,Cov,in,out);
    getBlock(Sig_22,Cov,in,in);
    getBlock(Mu_1,Mean,out);
    getBlock(Mu_2,Mean,in);

   // std::cout<< "just before condition" << std::endl;




    arma::mat Sigma = Sig_11 - Sig_12 * arma::inv(Sig_22) * Sig_21;
    arma::vec Mu    = Mu_1   + Sig_12 * arma::inv(Sig_22) * ( x - Mu_2 );

   // std::cout<< ".. here" << std::endl;

 /*   std::cout<< "x:     (" << x.n_rows      << " x " << x.n_cols << ")" << std::endl;
    std::cout<< "Mu_2   (" << Mu_2.n_rows   << " x " << Mu_2.n_cols << ")" << std::endl;
    std::cout<< "Mu_1   (" << Mu_1.n_rows   << " x " << Mu_1.n_cols << ")" << std::endl;
    std::cout<< "Sig_11 (" << Sig_11.n_rows << " x " << Sig_11.n_cols << ")" << std::endl;
    std::cout<< "Sig_22 (" << Sig_22.n_rows << " x " << Sig_22.n_cols << ")" << std::endl;
    std::cout<< "Mu:    (" << Mu.n_rows     << " x " << Mu.n_cols <<  ")" << std::endl;
*/

    Gaussian normal = *this;
    normal.SetMean(Mu);
    normal.SetCov(Sigma);

    return normal;
}

Gaussian::Gaussian(const arma::vec& Mean, const arma::mat& Cov):Mean(Mean),Cov(Cov),gen(boost::variate_generator<ENG,DIST>(ENG(),DIST(0,1))){

    D = Mean.n_elem;

    if(D > 1){
        try{
            A = arma::chol(this->Cov);
        }catch(std::runtime_error e){
            std::cout<< "chol did not work" << std::endl;
            this->Cov.eye() * 0.00001;
            A = arma::chol(this->Cov);
        }
        z.resize(Mean.n_elem);
        det = arma::det(Cov);
    }else{
        gen = boost::variate_generator<ENG,DIST>(ENG(),DIST((this->Mean)(0),(this->Cov)(0,0)));

    }
}


void Gaussian::getBlock(arma::mat& A_xx,arma::mat& A,const std::vector<std::size_t>& dim1,const std::vector<std::size_t>& dim2){
    for(unsigned int i = 0; i < dim1.size();i++){
        for(unsigned int j = 0; j < dim2.size();j++){
            A_xx(i,j) = A(dim1[i],dim2[j]);
        }
    }
}

void Gaussian::getBlock(arma::vec& Mu_xx, arma::vec Mu, const std::vector<std::size_t> &dim){
    for(unsigned int i = 0; i < dim.size();i++){
        Mu_xx(i) = Mu(dim[i]);
    }
}

void Gaussian::SetMean(const arma::vec& Mean){
    this->Mean = Mean;
    D = Mean.n_elem;
    z.resize(D);
}

void Gaussian::SetCov(const arma::mat& Cov){
    this->Cov = Cov;
    if(D > 1){
        try{
            A = arma::chol(this->Cov);
        }catch(std::runtime_error e){
            this->Cov.eye() * 0.00001;
            A = arma::chol(this->Cov);
        }
        det = arma::det(this->Cov);
    }
}

void Gaussian::fit(const arma::mat& data)
{
    if(data.size() > 0){
        D = data.n_rows;
        Mean = arma::mean(data);
        Cov = arma::cov(data);
        A = arma::chol(Cov);
        det = arma::det(Cov);
    }
}

double Gaussian::P(const arma::vec& X){
    if(Mean.n_elem > 1){
        //                                          (3 x 3) *  (3 x 1)
        double quadratic = arma::dot((X - Mean),arma::inv(Cov) * (X - Mean));
        return 1.0/(pow(2*M_PI,(double)D/2)*sqrt(det)) * exp(-0.5*quadratic);
    }else{
        double mu = Mean(0);
        double x = X(0);
        double var = Cov(0,0);
        return 1.0/(sqrt(2*M_PI) * sqrt(var)) * exp(-0.5*(x - mu)*(x - mu)/sqrt(var));
    }
}

arma::vec Gaussian::sample(){
    for(unsigned int i = 0; i < Mean.n_rows;i++){
        z[i] = gen();
    }
    //  (3x1) + (3x3) * (3x1)
    return Mean + (A*z);
}

