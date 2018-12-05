#include <chrono>
#include <iostream>
#include <algorithm>

double test_alloc(size_t n)
{
    auto t0 = std::chrono::system_clock::now();
    /* time to allocate + fill */
    char* ptr = (char*)std::malloc(n);
    std::fill(ptr, ptr + n, 0);
    auto t1 = std::chrono::system_clock::now();
    /* time fo fill */
    std::fill(ptr, ptr + n, 0);
    auto t2 = std::chrono::system_clock::now();
    t0 += static_cast<std::chrono::seconds>(ptr[0]);
    std::free(ptr);

    return std::chrono::duration_cast<std::chrono::duration<double>>((t1 - t0) - (t2 - t1)).count();
}

int main(int argn, char** argv)
{
    for (int i = 20; i < 33; i++) {
        size_t sz = size_t(1) << i;
        std::cout << sz / 1024.0 / 1024.0 << " MB, allocation time " << test_alloc(sz) << " sec. \n";
    }
}
