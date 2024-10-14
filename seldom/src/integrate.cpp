#include "decl.hpp"
#include "utils.hpp"
#include <unsupported/Eigen/SpecialFunctions>

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
            const auto u = 0.5 * (quad_x[i] + 1);
            return ((1 - odds * x - u) + odds * z).max(0).min(1).prod();
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
    const auto ss = util::rowvec_type<T>::NullaryExpr(
        quad_x.size(),
        [&](auto i) {
            const auto xi = quad_x[i];
            return selection_func(p_half * (xi+1), z, gamma, quad_x, quad_w); 
        }
    );
    return p_half * (ss * quad_w).sum();
}

template <class T>
T gauss_fd(
    T a,
    T t,
    const Eigen::Ref<const util::rowvec_type<T>>& quad_x,
    const Eigen::Ref<const util::rowvec_type<T>>& quad_w
)
{
    constexpr T sqrt_1_2pi = 0.5 * M_2_SQRTPI / M_SQRT2;
    const auto Phi = [&](auto x) {
        return 0.5 * (1 + (x / M_SQRT2).erf());
    };
    const auto fs = 1 - Phi((t - a) - quad_x);
    const auto phis = (-0.5 * (quad_x - (1-a)).square()).exp();
    const auto correction = sqrt_1_2pi * std::exp(0.5 - a);
    const auto numerator = (quad_w * fs * phis).sum() * correction;
    const auto denominator = 1 - 0.5 * (1 + std::erf(t * 0.5));
    return numerator / (denominator + (denominator <= 0));
}

} // namespace seldom

namespace sd = seldom;

void register_integrate(py::module_& m)
{
    m.def("selection_func", &sd::selection_func<double>); 
    m.def("integrate_selection_func", &sd::integrate_selection_func<double>);
    m.def("gauss_fd", &sd::gauss_fd<double>);
}