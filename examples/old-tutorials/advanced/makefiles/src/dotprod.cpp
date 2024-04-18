// Copyright 2016-2024 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
// ReFrame Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: BSD-3-Clause

#include <cassert>
#include <iostream>
#include <random>
#include <vector>

#ifndef ELEM_TYPE
#define ELEM_TYPE double
#endif

using elem_t = ELEM_TYPE;

template<typename T>
T dotprod(const std::vector<T> &x, const std::vector<T> &y)
{
    assert(x.size() == y.size());
    T sum = 0;
    for (std::size_t i = 0; i < x.size(); ++i) {
        sum += x[i] * y[i];
    }

    return sum;
}

template<typename T>
struct type_name {
    static constexpr const char *value = nullptr;
};

template<>
struct type_name<float> {
    static constexpr const char *value = "float";
};

template<>
struct type_name<double> {
    static constexpr const char *value = "double";
};

int main(int argc, char *argv[])
{
    if (argc < 2) {
        std::cerr << argv[0] << ": too few arguments\n";
        std::cerr << "Usage: " << argv[0] << " DIM\n";
        return 1;
    }

    std::size_t N = std::atoi(argv[1]);
    if (N < 0) {
        std::cerr << argv[0]
                  << ": array dimension must a positive integer: " << argv[1]
                  << "\n";
        return 1;
    }

    std::vector<elem_t> x(N), y(N);
    std::random_device seed;
    std::mt19937 rand(seed());
    std::uniform_real_distribution<> dist(-1, 1);
    for (std::size_t i = 0; i < N; ++i) {
        x[i] = dist(rand);
        y[i] = dist(rand);
    }

    std::cout << "Result (" << type_name<elem_t>::value << "): "
              << dotprod(x, y) << "\n";
    return 0;
}
