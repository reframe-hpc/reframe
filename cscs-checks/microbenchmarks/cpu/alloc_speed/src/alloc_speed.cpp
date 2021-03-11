#include <chrono>
#include <iostream>
#include <algorithm>

double test_alloc(size_t n, char c)
{
    /* time to allocate + fill */
    auto t0 = std::chrono::high_resolution_clock::now();
    /* memory is allocated using "malloc" since
       "std::unique_ptr<char[]> ptr(new char[n])"
       also creates the objects via "new[]" */
    char* ptr = (char*)std::malloc(n);
    std::fill(ptr, ptr + n, c);
    auto t1 = std::chrono::high_resolution_clock::now();

    double t_alloc_fill = std::chrono::duration_cast<std::chrono::duration<double>>(t1 - t0).count();
    /* prevent compiler optimizations */
    t_alloc_fill += static_cast<double>(*ptr);

    /* time to fill */
    t0 = std::chrono::high_resolution_clock::now();
    std::fill(ptr, ptr + n, c);
    t1 = std::chrono::high_resolution_clock::now();
    double t_fill = std::chrono::duration_cast<std::chrono::duration<double>>(t1 - t0).count();
    /* prevent compiler optimizations */
    t_fill += static_cast<double>(*ptr);

    std::free(ptr);

    return t_alloc_fill - t_fill;
}

int main(int argc, char** argv)
{
    for (size_t i = 20; i < 33; ++i)
    {
        std::cout << (1L << i) / static_cast<double>(1024.0*1024.0) << " MB, allocation time " << test_alloc(1L << i, 0) << " sec.\n";
    }

    return 0;
}
