#include <chrono>
#include <iostream>
#include <algorithm>
#include <memory>

double test_alloc(size_t n)
{
    auto t_start = std::chrono::system_clock::now();
    std::unique_ptr<char[]> ptr(new char[n]);

    /* time to allocate + fill */
    std::fill(ptr.get(), ptr.get() + n, 0);
    auto t_alloc_fill = std::chrono::system_clock::now();

    /* time too fill */
    std::fill(ptr.get(), ptr.get() + n, 0);
    auto t_alloc_two_fills = std::chrono::system_clock::now();

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
