#include <sys/time.h>
#include <iostream>
#include <algorithm>

/// Wall-clock time in seconds.
inline double wtime()
{
    timeval t;
    gettimeofday(&t, NULL);
    return double(t.tv_sec) + double(t.tv_usec) / 1e6;
}

double test_alloc(size_t n)
{
    double t0 = wtime();
    /* time to allocate + fill */
    char* ptr = (char*)std::malloc(n);
    std::fill(ptr, ptr + n, 0);
    double t1 = wtime();
    /* time fo fill */
    std::fill(ptr, ptr + n, 0);
    double t2 = wtime();
    std::free(ptr);

    return (t1 - t0) - (t2 - t1);
}

int main(int argn, char** argv)
{
    for (int i = 20; i < 33; i++) {
        size_t sz = size_t(1) << i;
        std::cout << sz / 1024.0 / 1024.0 << " MB, allocation time " << test_alloc(sz) << " sec. \n";
    }
}