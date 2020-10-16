// contributed by Sebastian Keller, CSCS, 2019-10

#include <iostream>
#include <cstdlib>
#include <chrono>
#include <stdlib.h>
#include <functional>
#include <vector>
#include <thread>
#include <memory>
#include <string>
#include <chrono>
#include <functional>

#include <sched.h>
#include <pthread.h>

template <class Callable>
size_t determine_loops(Callable const& func, double target_duration)
{
    // measure number of loops required
    size_t loops = 1;
    double elapsed = 0;
    do {
        loops *= 2;
        auto t0 = std::chrono::high_resolution_clock::now();
        func(loops);
        auto t1 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = t1 - t0;
        elapsed = duration.count();
    } while (elapsed < 0.1);
    loops *= target_duration/elapsed;

    return loops;
}

template <class Callable>
double timed_run(Callable const& func)
{
    auto t0 = std::chrono::high_resolution_clock::now();
    func();
    auto t1 = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> duration = t1 - t0;
    return duration.count();
}

class Buffer
{
public:
    Buffer() : data_size(0), data_(nullptr, free) {}

    explicit Buffer(size_t sz_) : data_size(sz_), data_(nullptr, free)
    {
        data_size = (sz_/sizeof(size_t)) * (sizeof(size_t));

        void* buf;
        if (posix_memalign(&buf, 32, data_size)) {
            std::cout << "alloc failed\n"; exit(1);
        } 
        // provide "free" as custom deleter
        data_ = std::unique_ptr<size_t[], decltype(free)*>((size_t*)buf, free);

        init();
    }

    Buffer(Buffer&& rhs) : data_(std::move(rhs.data_)), data_size(rhs.data_size) {}

    Buffer& operator=(Buffer&& rhs) {
        data_ = std::move(rhs.data_);
        data_size = rhs.data_size;
        return *this;
    }

    size_t* get() const { return data_.get(); }

private:

    void init()
    {
        for (size_t i = 0; i < data_size/sizeof(size_t); i++)
            data_[i] = 1;
    }

    std::unique_ptr<size_t[], decltype(free)*> data_;
    size_t data_size;
};


void set_affinity(std::thread& t, int i)
{
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(i, &cpuset);
    int rc = pthread_setaffinity_np(t.native_handle(),
                                    sizeof(cpu_set_t), &cpuset);
    if (rc != 0)
        std::cerr << "Error calling pthread_setaffinity_np: " << rc << "\n";
}

int get_affinity(std::thread& t)
{
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    int rc = pthread_getaffinity_np(t.native_handle(),
                                    sizeof(cpu_set_t), &cpuset);
    if (rc != 0)
        std::cerr << "Error calling pthread_setaffinity_np: " << rc << "\n";
    int ret;
    for (int i=0; i < 128; ++i)
        if (CPU_ISSET(i, &cpuset)) return i;

    return -1;
}

void update_stride(size_t*, size_t, size_t, size_t);


int main(int argc, char ** argv)
{
    if (argc != 4)
    {
        std::cout << "Usage: ./<prog_name> buffer_size stride nthreads\n"
                     "\n"
                     "buffer_size: in bytes\n"
                     "stride: in multiples of 8 bytes\n";
        exit(1);
    }

    size_t s = std::stoi(argv[1]);
    size_t stride = std::stoi(argv[2]);
    int nthreads = std::stoi(argv[3]);

    double time_per_run = 2.0;

    std::vector<Buffer> buf(nthreads);
    // initialize buffers (first touch)
    for (int i = 0; i < nthreads; ++i)
    {
        std::thread t([=, &buf](){
                std::this_thread::sleep_for(std::chrono::milliseconds(20));
                buf[i] = Buffer(s);
            });
        set_affinity(t, i);
        t.join();
    }

    auto func = [buf = buf[0].get(), s, stride](int loops) { update_stride(buf, s, loops, stride); };
    size_t loops = determine_loops(func, time_per_run);

    std::vector<std::function<void()>> tasks(nthreads);
    for (int i = 0; i < nthreads; ++i)
        tasks[i] = std::bind(&update_stride, buf[i].get(), s, loops, stride);

    std::vector<std::thread> workers(nthreads);
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < nthreads; ++i)
    {
        workers[i] = std::thread(tasks[i]);
        set_affinity(workers[i], i);
    }

    for (auto& t: workers)
        t.join();

    auto t1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = t1 - t0; 
    double time = elapsed.count();

    size_t load_and_store = 2;
    size_t bytes_moved = nthreads * s * load_and_store * loops / stride;
    std::cout << "buffer size: " << s/1024 << "Kb, bandwidth: "
              << double(bytes_moved)/(1024*1024*1024) / time
              << " GB/s" << std::endl;
}



void update_stride(size_t* src, size_t nbytes, size_t iterations, size_t stride)
{
    while (iterations--)
    {
        for (size_t i = 0; i < nbytes/sizeof(size_t); i+=stride)
        {
            src[i]++; // load & store
        }
    }
}
