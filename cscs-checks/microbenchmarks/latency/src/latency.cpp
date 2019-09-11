// contributed by Sebastian Keller, CSCS, 2019-7

#include <vector>
#include <iostream>
#include <algorithm>
#include <chrono>
#include <random>

#include <thread>
#include <pthread.h>

#ifndef PAGESIZE
#define PAGESIZE 4096
#endif

#ifndef CACHELINESIZE
#define CACHELINESIZE 64
#endif

// only used for runtime estimation
#define L1 32768
#define L2 262144
#define L3 25600*1024

#ifndef GHZ
#define GHZ 3.3
#endif

/**********************************************/

struct alignas(CACHELINESIZE) CacheLine
{
    CacheLine* payload;
    char padding[CACHELINESIZE - sizeof(CacheLine*)];
};

std::vector<unsigned> random_path(unsigned);
std::vector<unsigned> random_grouped_path(unsigned, unsigned);

void convert_to_pointers(CacheLine*, std::vector<unsigned> const&);
size_t chainload(CacheLine*, size_t reps);
size_t estimate_reps(unsigned sz, unsigned l1, unsigned l2, unsigned l3);
void set_affinity(pthread_t t, int i);

template <class F, class ...Args>
double time_function(F func, Args&& ...args)
{
    auto t1 = std::chrono::high_resolution_clock::now();
    auto endpoint = func(std::forward<Args>(args)...);
    auto t2 = std::chrono::high_resolution_clock::now();

    volatile auto result = endpoint;
    return std::chrono::duration<double>(t2-t1).count();
}


/**********************************************/

int main(int argc, char ** argv)
{
    set_affinity(pthread_self(), 0);
    bool simulate_large_pages = false;

    if (argc < 2)
    {
        std::cout << "Measure latencies for various levels of "
                  << "the memory hierarchy (buffer sizes)" << std::endl;
        std::cout << "Usage: latency <buffer sizes>" << std::endl;
    }

    for (int iarg = 1; iarg < argc; ++iarg)
    {
        unsigned buffer_size_inp = std::stoi(argv[iarg]);

        unsigned buffer_size = (buffer_size_inp / PAGESIZE + 1) * PAGESIZE;

        unsigned group_size;
        if (simulate_large_pages) {
            unsigned npages = buffer_size / PAGESIZE;
            // use 48 out of 64 entries in TLB-L1
            unsigned ngroups = npages / 48 + 1;
            unsigned pages_per_group = npages / ngroups;
            group_size = pages_per_group * PAGESIZE;
            buffer_size = group_size * ngroups;
        }
        else
            group_size = buffer_size;

        unsigned num_elements = buffer_size / sizeof(CacheLine);
        unsigned group_elements = group_size / sizeof(CacheLine);

        size_t reps = estimate_reps(buffer_size, L1, L2, L3);

        CacheLine* chain;
        if (posix_memalign(reinterpret_cast<void**>(&chain),
                           PAGESIZE, num_elements * sizeof(CacheLine)))
            throw std::bad_alloc();

        std::vector<unsigned> path = random_grouped_path(num_elements, group_elements);
        convert_to_pointers(chain, path);

        double test = time_function(chainload, chain, reps/10);
        reps = reps * 0.25/(test*10);

        double duration = 0;
        duration += time_function(chainload, chain, reps);

        double ns_per_load = duration / reps * 1e9;
        double clocks = GHZ * ns_per_load;

        std::cout << "latency (ns) for input size " << buffer_size_inp << ": " << ns_per_load
                  << " clocks: " << clocks << " size " << buffer_size
                  << " ngroups: " << buffer_size/group_size << std::endl;

        std::free(chain);
    }
}


/**********************************************/


void convert_to_pointers(CacheLine* start, std::vector<unsigned> const& path)
{
    for (unsigned i = 0; i < path.size(); ++i)
        start[i].payload = start + path[i];
}


size_t chainload(CacheLine* next, size_t reps)
{
    for (size_t i = 0; i < reps; i++)
        next = next->payload;

    return reinterpret_cast<size_t>(next);
}


// create a random permutation of numbers [0,length-1]
std::vector<unsigned> random_permutation(unsigned length)
{
    static unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    static auto gen = std::default_random_engine(seed);

    std::vector<unsigned> ret(length);
    unsigned var = 0;

    std::generate(ret.begin(), ret.end(), [&var]() { return var++; } );
    std::shuffle(ret.begin(), ret.end(), gen);

    return ret;
}


// convert any permutation visit_sequence to a path
// --
// visit_sequence is the order in which we want to visit the nodes
// --
// Each element of the path contains a "pointer" (index) to the next
// element to visit according to the visit sequence.
// For example, a visit_sequence of 12034 will produce the path 32041.
// --
// Mathematically: A path is equivalent to a permutation of [0, length-1]
// that contains no valid permutation in any subset
// [0,n] for n < length-1
std::vector<unsigned> make_path(std::vector<unsigned> const& visit_sequence)
{
    std::vector<unsigned> path(visit_sequence.size());

    int s = visit_sequence[0];
    for (int i = 1; i < path.size(); ++i)
    {
        path[s] = visit_sequence[i];
        s = visit_sequence[i];
    }

    path[s] = visit_sequence[0];

    return path;
}


// Create a random path through positions [0, length-1]
// that is guaranteed to visit all elements
std::vector<unsigned> random_path(unsigned length)
{
    std::vector<unsigned> ret(length, 0);
    std::vector<unsigned> permutation = random_permutation(length);

    return make_path(permutation);
}

// Create a random path through length/groupsize groups
// visit all elements randomly in each group, then randomly
// move to next group
// e.g. pick a random page, visit randomly all elements in the page,
// then pick next page randomly
std::vector<unsigned> random_grouped_path(unsigned length, unsigned groupsize)
{
    unsigned ngroups = length/groupsize;

    auto group_perm = random_permutation(ngroups);
    std::vector<unsigned> grouped_seq(length);
    for (int p = 0; p < ngroups; ++p)
    {
        unsigned page_nr = group_perm[p] * groupsize;

        std::vector<unsigned> intra_page_perm = random_permutation(groupsize);
        for (int i = 0; i < groupsize; ++i)
            grouped_seq[p*groupsize + i] = page_nr + intra_page_perm[i];
    }

    return make_path(grouped_seq);
}


size_t estimate_reps(unsigned sz, unsigned l1, unsigned l2, unsigned l3)
{
    const double l1_lat_ns = 1;
    const double l2_lat_ns = 3;
    const double l3_lat_ns = 10;
    const double mem_lat_ns = 100;

    const double target_time = 0.25;

    if (sz < l1) return size_t(target_time / l1_lat_ns * 1e9);
    if (sz < l2) return size_t(target_time / l2_lat_ns * 1e9);
    if (sz < l3) return size_t(target_time / l3_lat_ns * 1e9);
    return size_t(target_time / mem_lat_ns * 1e9);
}


void set_affinity(pthread_t t, int i)
{
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(i, &cpuset);
    int rc = pthread_setaffinity_np(t, sizeof(cpu_set_t), &cpuset);
    if (rc != 0)
        std::cerr << "Error calling pthread_setaffinity_np: " << rc << "\n";
}
