#pragma once
#include <Eigen/Core>

namespace seldom {
namespace util {

template <class T>
using rowvec_type = Eigen::Array<T, 1, Eigen::Dynamic, Eigen::RowMajor>;

} // namespace util
} // namespace seldom