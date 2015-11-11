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


// Statistics

#include <mlpack/methods/gmm/gmm.hpp>

// Armadillo


#include "gaussian.h"

#include <armadillo>

class GMM {

public :

    GMM();

    GMM(std::string path_to_parameter_folder);

    GMM(const arma::vec &weights,const std::vector<arma::vec>& Mu,const std::vector<arma::mat>& Sigma);

    void likelihood(const arma::mat& X,arma::vec& y);

    double nlikelihood(const arma::vec& x);

    void P(const arma::mat& X,arma::vec& L);

    double getPriors(std::size_t i);

    arma::vec& getMu(std::size_t i);

    void setMu(const arma::vec& mu, std::size_t k);

    arma::mat& getSigma(std::size_t i);

    void expection(arma::vec& x);

    void condition(GMM& gmm,arma::vec& x,const std::vector<std::size_t>& in,const std::vector<std::size_t>& out);

    void sample(arma::mat& X);

    void setParam(const arma::vec &weights,const std::vector<arma::vec>& Mu,const std::vector<arma::mat>& Sigma);

    void load(const std::string& file);

    void clear();

    void set_name(const std::string& name);

    void print();

    std::size_t K;
    std::size_t D;
    std::string name;


    boost::mt19937 generator;
    boost::random::discrete_distribution<> dist;
    Gaussian gaussian;


    mlpack::gmm::GMM<double> gmm;

private:

    std::vector<arma::mat> A;

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
           gmms[i].set_name(paths_to_gmms[i].filename().string());
       }

       return true;
   }

};


#endif
