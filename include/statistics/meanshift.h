/*
 * meanshift.h
 *
 *  Created on: Dec 11, 2012
 *      Author: guillaume
 */

#ifndef MEANSHIFT_H_
#define MEANSHIFT_H_

#include <vector>
#include <utility>
#include <set>
#include <algorithm>
#include <nanoflann.hpp>
#include <statistics/distributions/distributions.h>
#include <armadillo>


namespace mean_shift{

using namespace std;
using namespace nanoflann;

struct PointCloud
{

    const arma::mat& pts;

    PointCloud(const arma::mat& pts):pts(pts){

    }

    arma::colvec3 getPoint(size_t index){
        return pts.row(index).st();
	}

	// Must return the number of data points
	inline size_t kdtree_get_point_count() const { return pts.n_rows; }

	// Returns the distance between the vector "p1[0:size-1]" and the data point with index "idx_p2" stored in the class:
	inline double kdtree_distance(const double *p1, const size_t idx_p2,size_t size) const
    {
        return sqrt((p1[0] - pts(idx_p2,0))*(p1[0] - pts(idx_p2,0)) + (p1[1] - pts(idx_p2,1))*(p1[1] - pts(idx_p2,1)) + (p1[2] - pts(idx_p2,2))*(p1[2] - pts(idx_p2,2)));
	}

	inline double kdtree_get_pt(const size_t idx, int dim) const
	{
		if (dim==0){
			return pts(idx,0);
		}else if (dim==1){
			return pts(idx,1);
		}else{
			return pts(idx,2);
		}
	}

	template <class BBOX>
	bool kdtree_get_bbox(BBOX &bb) const { return false; }

};

// construct a kd-tree index:
typedef KDTreeSingleIndexAdaptor<L2_Simple_Adaptor<double, PointCloud > , 	PointCloud, 3 /* dim */ > my_kd_tree_t;


class MeanShift_Parameters{
public:

    MeanShift_Parameters(){
        //scale                   = 100;  // convert meters to centimeters
        bandwidth               = 1.0/(0.005 * 0.005);
        max_iterations          = 100;
        convergence_threashold  = 0.001;
        min_merge_distance      = 0.005;

    }

    double      bandwidth;
    double      convergence_threashold;
    double      scale;
    int         max_iterations;
    double      min_merge_distance;
};



class MeanShift{

public:

    MeanShift(const arma::mat& points,const MeanShift_Parameters& mean_shift_parameters);

    ~MeanShift();

    void update();

    void one_step_update();

    void set_initial_center_guess(const arma::mat &centroids);

    void merge_centroids(double min_distance);

private:

    void compute_gradients();


private:


    void find_index_to_merge(arma::colvec3& c,arma::mat& points,double threashod);

    inline void vec2arr(const arma::colvec3& v1,double* arr){
        arr[0]  = v1(0);
        arr[1]  = v1(1);
        arr[2]  = v1(2);
    }


public:

      arma::mat                             centroids;
      arma::mat                             modes;

private:

    PointCloud                              cloud;
    const arma::mat&                        points;
    arma::mat                               centroids_tmp;

    std::vector<arma::colvec3>              m;

    std::size_t                             num_centroids;
    std::size_t                             iterations;
    std::size_t                             max_iterations;

    double                                  sum_w,w;
    double                                  bandwidth;
    double                                  difference;
    double                                  convergence_threashold;
    double                                  scale;

    my_kd_tree_t*                           index;
    nanoflann::SearchParams                 params;
    std::vector<std::pair<size_t,double> >  ret_matches;
    double                                  query_pt[3];

};

}


#endif /* MEANSHIFT_H_ */
