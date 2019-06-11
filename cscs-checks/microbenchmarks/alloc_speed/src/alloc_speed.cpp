#include <chrono>
#include <iostream>
#include <algorithm>

double test_alloc(size_t n)
{
    auto t_start = std::chrono::system_clock::now();

    /* time to allocate + fill */
    /* memory is allocated using "malloc" since
       "std::unique_ptr<char[]> ptr(new char[n])"
       also creates the objects via "new[]" */
    char* ptr = (char*)std::malloc(n);
    std::fill(ptr, ptr + n, 0);
    auto t_alloc_fill = std::chrono::system_clock::now();

    /* time to fill */
    std::fill(ptr, ptr + n, 0);
    auto t_alloc_two_fills = std::chrono::system_clock::now();

    /* prevent compiler optimizations */
    t_start += std::chrono::nanoseconds(ptr[0]);

    std::free(ptr);

    return std::chrono::duration_cast<std::chrono::duration<double>>((t_alloc_fill - t_start) - (t_alloc_two_fills - t_alloc_fill)).count();
}

int main(int argc, char** argv)
{
    for (size_t i = 20; i < 33; ++i)
    {
        std::cout << (1L << i) / static_cast<double>(1024.0*1024.0) << " MB, allocation time " << test_alloc(1L << i) << " sec.\n";
    }

    return 0;
}
