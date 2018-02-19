#include <boost/python.hpp>

char const* greet()
{
   return "hello, world";
}

BOOST_PYTHON_MODULE(hello_boost_python)
{
    using namespace boost::python;
    def("greet", greet);
}
