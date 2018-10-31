#include <numeric>
using namespace std;

extern"C" void do_smth_with_std(float* x, int n, int* res)
{
    *res = 0;
    *res = accumulate(x, x+n, 0);
}
