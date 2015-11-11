#ifndef STATS_DISTANCE_METRIC_H_
#define STATS_DISTANCE_METRIC_H_

#include <mlpack/core/metrics/lmetric.hpp>


namespace mlpack {
namespace metric {


template<int Power, bool TakeRoot = true>
class LWMetric {
public:

     LWMetric(arma::colvec& w):w(w){ }
  //  LWMetric(){}

    /**
     * Computes the distance between two points.
     */
    template<typename VecType1, typename VecType2>
    double Evaluate(const VecType1& a, const VecType2& b);

    template<typename VecType1, typename VecType2>
    double Evaluate(const VecType1& a, const VecType2& b,std::size_t i);

    std::string ToString() const;
private:

   arma::colvec& w;

};

/***
 * The squared Euclidean (L2) distance.
 */
typedef LWMetric<2, false> WSquaredEuclideanDistance;

}
}

// Include implementation.
#include <statistics/impl/distance_metric_impl.hpp>

#endif
