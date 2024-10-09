#include "decl.hpp"
#include "utils.hpp"

namespace seldom {

template <class T>
T selection_func(
    T x,
    const Eigen::Ref<const util::rowvec_type<T>>& z,
    T gamma,
    const Eigen::Ref<const util::rowvec_type<T>>& quad_x,
    const Eigen::Ref<const util::rowvec_type<T>>& quad_w
)
{
    const auto odds = gamma / (1-gamma);
    const auto fs = util::rowvec_type<T>::NullaryExpr(
        quad_x.size(), 
        [&](auto i) {
            return ((1 - odds * x - 0.5 * (quad_x[i] + 1)) + odds * z).max(0).min(1).prod();
        }
    );
    return 0.5 * (fs * quad_w).sum();
}

template <class T>
T integrate_selection_func(
    T p,
    const Eigen::Ref<const util::rowvec_type<T>>& z,
    T gamma,
    const Eigen::Ref<const util::rowvec_type<T>>& quad_x,
    const Eigen::Ref<const util::rowvec_type<T>>& quad_w
)
{
    const auto p_half = 0.5 * p;
    return p_half * util::rowvec_type<T>::NullaryExpr(
        quad_x.size(),
        [&](auto i) {
            const auto xi = quad_x[i];
            const auto wi = quad_w[i];
            return wi * selection_func(p_half * (xi+1), z, gamma, quad_x, quad_w); 
        }
    ).sum();
}

} // namespace seldom

namespace sd = seldom;

void register_integrate(py::module_& m)
{
    m.def("selection_func", &sd::selection_func<double>); 
    m.def("integrate_selection_func", &sd::integrate_selection_func<double>);
}