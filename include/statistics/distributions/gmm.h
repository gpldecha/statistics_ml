#ifndef GMM_H_
#define GMM_H_

// Boost
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/normal_distribution.hpp>
#include <boost/random/uniform_real.hpp>
#include <boost/random/variate_generator.hpp>
#include <boost/random/discrete_distribution.hpp>
#include <boost/range.hpp>
#include <boost/array.hpp>
#include <boost/filesystem.hpp>


// STL

#include <vector>


// Armadillo


#include "gaussian.h"

#include <armadillo>

namespace stats{


class Load_param{

public:

    class scale{
    public:

        scale(){
            dim = 0;
            min_d = 0;
            max_d = 1;
            target_min = 0;
            target_max = 1;
            bscale = false;
        }

        void print() const{
            std::cout<< "== scale == " << std::endl;
            std::cout<< "bscale: " << bscale << std::endl;
            std::cout<< "dim:   " << dim << std::endl;
            std::cout<< "min_d: " << min_d << std::endl;
            std::cout<< "max_d: " << max_d << std::endl;
            std::cout<< "t_min: " << target_min << std::endl;
            std::cout<< "t_max: " << target_max << std::endl;
        }

        std::size_t dim;
        double min_d,max_d;
        double target_min,target_max;
        bool bscale;

    };

public:

    void load_scale(const std::string path_param);

public:

    scale scale_;
    arma::mat tmp;

};


class GMM {

public :

    GMM();

    GMM(std::string path_to_parameter_folder);

    GMM(const arma::colvec &weights,const std::vector<arma::vec>& Mu,const std::vector<arma::mat>& Sigma);

    const arma::colvec& get_weigts() const;

    const std::vector<arma::colvec>& get_means() const;

    const std::vector<arma::mat>& get_covariances() const;

    void set_prior(const arma::colvec &weights);

    void set_prior(const std::size_t i,const double value);

    void set_mu(const std::size_t i,const arma::vec& mu);

    void set_covariance(const std::size_t i,const arma::mat& covariance);

    void likelihood(const arma::mat& X,arma::vec& y);

    void expection(arma::colvec& x) const;

    void sample(arma::mat& X);

    void load(const std::string& file);

    void clear();

    void print() const;

    std::size_t K;
    std::size_t D;
    std::string name;
    std::vector<std::size_t>  in, out;

    boost::mt19937 generator;
    boost::random::discrete_distribution<> dist;

protected:

    arma::colvec              pi;
    std::vector<arma::colvec> Means;
    std::vector<arma::mat>    Covariances;




private:

    std::vector<arma::mat> A;

};
/**
 * @brief The cGMM class, conditional Gaussian mixture model
 *
 *
 */

class cGMM{

public:

    /**
     * @brief condition : conditions the elements of the Gaussian functions which do not depend
     *                    on the values of the input vector. The conditioned covariances are computed
     *                    here : $\Sigma_{a|b}$
     *
     *                    This function has to be called first and only once !!!
     *
     * @param gmm_in    : GMM which is to be conditioned
     * @param in        : dimensions which we condition on
     * @param out       : remaining dimensions of the GMM
     */
    void condition(const GMM& gmm_in,const std::vector<std::size_t>& in,const std::vector<std::size_t>& out);

    /**
     * @brief condition : computes the values of the conditioned parameters \mu_{a|b}, \pi_{a|b}.
     *
     *                    This function will be called in the for loop of your controller.
     *
     * @param x_in      : x_b (M x 1), M dimensional vector which we condition the gmm on
     * @param gmm_in    : GMM which is to be conditioned
     */
    void condition(const arma::colvec& x_in,const GMM& gmm_in);

    void print(const std::string& p="") const;

private:

    std::vector<Gaussian_c>   gaussian_c;

public:

    GMM                        gmm_c;

};


class GMM_Load{
public:

    static bool get_folders_in_dir(const std::string& path_to_parameters,std::vector<boost::filesystem::path>& paths_to_gmm){

        namespace fs = boost::filesystem;
        fs::path someDir(path_to_parameters);
        fs::directory_iterator end_iter;

        if(!fs::exists(someDir) && !fs::is_directory(someDir)){
            std::cerr << "Gmm_classifier::get_folders_in_dir no such directory: " << path_to_parameters << std::endl;
            return false;
        }

        if ( fs::exists(someDir) && fs::is_directory(someDir))
        {
            for( fs::directory_iterator dir_iter(someDir) ; dir_iter != end_iter ; ++dir_iter)
            {
                paths_to_gmm.push_back(dir_iter->path());
            }
        }
        return true;
    }


    static bool load_gmms(const std::string& path_to_parameters,std::vector<GMM>& gmms){
        std::vector<boost::filesystem::path> paths_to_gmms;

        if(!get_folders_in_dir(path_to_parameters,paths_to_gmms)){
            return false;
        }

        gmms.resize(paths_to_gmms.size());

        for(std::size_t i = 0; i < paths_to_gmms.size();i++){
            gmms[i].load(paths_to_gmms[i].string() + "/");
        }

        return true;
    }

};

}

#endif
