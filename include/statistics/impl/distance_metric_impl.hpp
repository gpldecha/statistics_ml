#ifndef STATS_DISTANCE_METRIC_IMPL_HPP_
#define STATS_DISTANCE_METRIC_IMPL_HPP_

// In case it hasn't been included.
#include <statistics/metric/distance_metric.hpp>

namespace mlpack {
namespace metric {

// Unspecialized implementation.  This should almost never be used...
template<int Power, bool TakeRoot>
template<typename VecType1, typename VecType2>
double LWMetric<Power, TakeRoot>::Evaluate(const VecType1& a,
                                             const VecType2& b)
{
  double sum = 0;
  for (size_t i = 0; i < a.n_elem; i++)
    sum += pow(w(i) * fabs(a[i] - b[i]), Power);

  if (!TakeRoot) // The compiler should optimize this correctly at compile-time.
    return sum;

  return pow(sum, (1.0 / Power));
}

// String conversion.
template<int Power, bool TakeRoot>
std::string LWMetric<Power, TakeRoot>::ToString() const
{
  std::ostringstream convert;
  convert << "LMetric [" << this << "]" << std::endl;
  convert << "  Power: " << Power << std::endl;
  convert << "  TakeRoot: " << (TakeRoot ? "true" : "false") << std::endl;
  return convert.str();
}



template<>
template<typename VecType1, typename VecType2>
double LWMetric<2, true>::Evaluate(const VecType1& a, const VecType2& b, std::size_t i)
{
  return sqrt(accu(w(i) % square(a - b)));
}

template<>
template<typename VecType1, typename VecType2>
double LWMetric<2, true>::Evaluate(const VecType1& a, const VecType2& b)
{
  return sqrt(accu(square(a - b)));
}

template<>
template<typename VecType1, typename VecType2>
double LWMetric<2, false>::Evaluate(const VecType1& a, const VecType2& b, std::size_t i)
{
  return accu(w(i) * square(a - b));
}

template<>
template<typename VecType1, typename VecType2>
double LWMetric<2, false>::Evaluate(const VecType1& a, const VecType2& b)
{
  return accu(square(a - b));
}


} // namespace metric
} // namespace mlpack

#endif
