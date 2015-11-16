#include <statistics/distributions/gmm.h>
#include <ros/ros.h>

int main(int argc,char** argv){


    std::cout<< "=== test GMM and GMC === " << std::endl;

    ros::init(argc, argv, "test_statistics");
    ros::NodeHandle nh;


    std::string path_parameters = "/home/guillaume/MatlabWorkSpace/peg_in_hole_RL/PolicyModelSaved/PolicyModel_txt/gmm_xhu";


    stats::GMM gmm(path_parameters);
/*
    stats::GMM gmm_c;
    std::vector<std::size_t> in = {{0,1,2,3}};
    std::vector<std::size_t> out = {{4,5,6}};
    arma::vec x(4);
    x(0) = 0.1765;
    x(1) = 0.3907;
    x(2) = 0.0705;
    x(3) = -1.9986;


    std::cout<< "w(0): " << (gmm.gmm.Weights())[0] << std::endl;
    gmm.gmm.Means()[0].print("Mu(0)");
    gmm.gmm.Covariances()[0].print("Covariance[0]");




    gmm_c.gmm = mlpack::gmm::GMM<double>(gmm.K,3);

    std::cout<< "--- condition---" << std::endl;

    std::cout<< "in : " << in.size() << std::endl;
    std::cout<< "out: " << out.size() << std::endl;

    gmm.condition(gmm_c,x,in,out);*/

    /*std::cout<< "w(0): " << (gmm_c.gmm.Weights())[114] << std::endl;
    gmm_c.gmm.Means()[114].print("Mu(0)");
    gmm_c.gmm.Covariances()[114].print("Covariance[0]");


    arma::vec e; e.resize(3);
*/
   // gmm_c.print();

   // gmm_c.expection(e);

    //e.print("expectation");



    std::cout<< "=== end of test === " << std::endl;



    return 0;
}
