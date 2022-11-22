// Copyright 2016-2022 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
// ReFrame Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: BSD-3-Clause

#include <iomanip>
#include <iostream>
#include <mutex>
#include <thread>
#include <vector>


#ifdef SYNC_MESSAGES
std::mutex hello_mutex;
#endif


void greetings(int tid)
{
#ifdef SYNC_MESSAGES
    const std::lock_guard<std::mutex> lock(hello_mutex);
#endif
    std::cout << "[" << std::setw(2) << tid << "] " << "Hello, World!\n";
}


int main(int argc, char *argv[])
{
    int nr_threads = 1;
    if (argc > 1) {
        nr_threads = std::atoi(argv[1]);
    }

    if (nr_threads <= 0) {
        std::cerr << "thread count must a be positive integer\n";
        return 1;
    }

    std::vector<std::thread> threads;
    for (auto i = 0; i < nr_threads; ++i) {
        threads.push_back(std::thread(greetings, i));
    }

    for (auto &t : threads) {
        t.join();
    }

    return 0;
}
