#include "decl.hpp"

PYBIND11_MODULE(seldom_core, m) {
    auto m_integrate = m.def_submodule("integrate", "Integrate submodule.");
    register_integrate(m_integrate);
}