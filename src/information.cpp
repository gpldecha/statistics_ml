#include <statistics/information.h>
#include <math.h>

Smooth::Smooth(){
    bFirstExp=true;
    s=0;
}


Information::Information(){
    qHu.clear();
    qHl.clear();
    qHo.clear();
}


double Information::monte_carlo_entropy(const GMM& gmm) {
    return -1;
}

double Information::upper_bound_entropy(GMM &gmm)   {
    double HU = 0;
    for(unsigned int i = 0; i < gmm.K; i++){
        HU = HU  - gmm.getPriors(i)*log(gmm.getPriors(i)) + 0.5*gmm.getPriors(i) *log(std::pow(2*M_PI*exp(1),(double)gmm.D) * arma::det(gmm.getSigma(i)))  ;
        //log(std::pow(2*M_PI*exp(1),(double)gmm.D) * arma::det(gmm.getSigma(i)))

    }
    HU = checkForNan(HU,qHu);
    return HU;
}

double Information::lower_bound_entropy(GMM& gmm) {
    double HL = 0;
    double tmp = 0;
    for(unsigned int i = 0; i < gmm.K;i++){
        tmp = 0;
        for(unsigned int j = 0; j < gmm.K;j++){
            tmp = tmp + gmm.getPriors(j) *  mvnpdf(gmm.getMu(i),gmm.getMu(j),gmm.getSigma(i) + gmm.getSigma(j));
        }
        tmp = log(tmp);
        HL = HL + gmm.getPriors(i) * tmp;
    }
    HL = -1*HL;
    HL = checkForNan(HL,qHl);

    return HL;
}

double Information::checkForNan(double x,std::deque<double>& q){
    if(q.size() == 0){
        if(std::isinf(x) || std::isnan(x)){
             q.push_back(0);
             return 0;
        }else{
             q.push_back(x);
             return x;
        }
    }else{
        if(std::isinf(x) || std::isnan(x)){
            return q.front();
        }else{
            q.pop_front();
            q.push_back(x);
            return x;
        }
    }

}

 double Information::entropy_gmm_one_gaussian(const arma::mat& C){
        double H0 = 0.5* log(std::pow(2*M_PI*exp(1),(double)C.n_rows) * arma::det(C));
        return checkForNan(H0,qHo);
 }



 double Information::scaled_sum_eigenvalue(const arma::mat& covariance,double min, double max){
     arma::vec eigval;
     eig_sym(eigval, covariance);
     return rescale(arma::sum(eigval),min,max,0,1);
 }

 double Information::sum_eigevalues(const arma::mat& covariance){
     arma::vec eigval;
     eig_sym(eigval, covariance);
     return arma::sum(eigval);
 }




