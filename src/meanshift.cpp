/*
 * meanshift.cpp
 *
 *  Created on: Dec 11, 2012
 *      Author: guillaume
 */

#include "statistics/meanshift.h"
#include <algorithm>
#include <boost/lexical_cast.hpp>

namespace mean_shift{


MeanShift::MeanShift(const arma::mat& points, const MeanShift_Parameters& mean_shift_parameters):
  cloud(points),
  points(points)
{
    bandwidth                   = mean_shift_parameters.bandwidth;
    convergence_threashold      = mean_shift_parameters.convergence_threashold;
    max_iterations              = mean_shift_parameters.max_iterations;
    scale                       = mean_shift_parameters.scale;

    index               =  new my_kd_tree_t(3, cloud, KDTreeSingleIndexAdaptorParams(10) );
}

MeanShift::~MeanShift(){
    if(index != NULL){
        delete index;
        index = NULL;
    }
}

void MeanShift::set_initial_center_guess(const arma::mat& centroids){
    this->centroids     = centroids;
    centroids_tmp       = centroids;
    num_centroids       = centroids.n_rows;
    m.resize(num_centroids);
}

void MeanShift::update(){

    index->buildIndex();
	iterations=0;

    while(iterations < max_iterations){
        compute_gradients();
        difference = arma::norm(centroids - centroids_tmp,2);
        if(difference < convergence_threashold){
			break;
		}
		iterations++;
	}
}

void MeanShift::one_step_update(){
    index->buildIndex();
    compute_gradients();
}

void MeanShift::compute_gradients(){

    arma::colvec3 x;
    arma::colvec3 center;
    sum_w = 0;
    for(std::size_t c = 0; c < num_centroids;c++){

        center = centroids.row(c).st();

        vec2arr(center,query_pt);
        index->radiusSearch(&query_pt[0],bandwidth*2, ret_matches, params);

        m[c].zeros();
        for(std::size_t i = 0; i < ret_matches.size();i++){
            x       = cloud.getPoint(ret_matches[i].first);
            w       = exp( -0.5 * bandwidth * arma::dot(center - x,center - x) );
            m[c]    = m[c] + w * x;
            sum_w   += w;
        }
        m[c] = m[c]/sum_w;
        sum_w = 0;
    }

    for(std::size_t c = 0; c < num_centroids;c++){
         centroids.row(c) = m[c].st();
    }

}

void MeanShift::merge_centroids(double min_distance){
    modes = centroids;
    arma::colvec3 p;
    std::size_t i = 0;
    while(i != modes.n_rows){
        p = modes.row(i).st();
        find_index_to_merge(p,modes,min_distance);
        i++;
    }
    // quick stupied fix
    i = 0;
    while(i != modes.n_rows){
        p = modes.row(i).st();
        find_index_to_merge(p,modes,min_distance);
        i++;
    }
}

void MeanShift::find_index_to_merge(arma::colvec3& c,arma::mat& points, double threashod){
    std::vector<std::size_t> index_keep;
    std::vector<std::size_t> index_merge;

    for(std::size_t i = 0; i < points.n_rows;i++){
        if(arma::norm(c - points.row(i).st(),2) < threashod){
            index_merge.push_back(i);
        }else{
            index_keep.push_back(i);
        }
    }

    if(index_merge.size() != 0){
        arma::colvec3 new_point;
        arma::mat new_points(index_keep.size()+1,3);
        for(std::size_t i = 0; i < index_merge.size();i++){
            new_point = new_point + points.row(index_merge[i]).st();
        }
        for(std::size_t i = 0; i < index_keep.size();i++){
            new_points.row(i) = points.row(index_keep[i]);
        }
        new_points.row(index_keep.size()) = new_point.st();
        points = new_points;
    }
}

}


 /*
void MeanShift::compute_covariances(){
    covariances.resize(modes.n_rows);
    priors.resize(modes.n_rows);
   N.resize(modes.n_rows);


    std::cout<< "modes: (" << modes.n_rows << " x " << modes.n_cols << ")" <<std::endl;
    std::size_t ii;
    for(std::size_t k = 0; k < modes.n_rows;k++){
        query_pt[0] = modes(k,0);
        query_pt[1] = modes(k,1);
        query_pt[2] = modes(k,2);

        ret_matches.clear();
        index->radiusSearch(&query_pt[0],bandwidth*2, ret_matches, params);

        covariances[k].resize(3,3);


        for(std::size_t j = 0; j < ret_matches.size();j++){
            ii                  = ret_matches[j].first;
            covariances[k]      = covariances[k] + (points.row(ii) - modes.row(k)).st() * (points.row(ii) - modes.row(k));
        }
        priors(k) = static_cast<double>(ret_matches.size());
        N[k]      = static_cast<double>(ret_matches.size()) + 1;
    }

    for(std::size_t k = 0; k < covariances.size();k++){
        covariances[k] = (covariances[k])/N[k];
        //covariances[k] = covariances[k] + I;
        //covariances[k] = 0.5*(covariances[k] + covariances[k].st());
    }
    priors = priors / arma::sum(priors);
}
*/
