/*
 * Copyright (C) 2002-2017 the Network-Based Computing Laboratory
 * (NBCL), The Ohio State University.
 *
 * Contact: Dr. D. K. Panda (panda@cse.ohio-state.edu)
 *
 * For detailed copyright and licensing information, please refer to the
 * copyright file COPYRIGHT in the top level OMB directory.
 */
#ifndef OSU_COLL_H
#define OSU_COLL_H 1
#endif

#ifndef OSU_PT2PT_H
#define OSU_PT2PT_H 1
#endif

#include <assert.h>
#include <mpi.h>
#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <stdint.h>
#include <getopt.h>
#include <pthread.h>
#include <inttypes.h>
#include <sys/time.h>
#include <limits.h>



#ifdef _ENABLE_CUDA_
#include "cuda.h"
#include "cuda_runtime.h"
#endif

#define MIN(a,b) ((a)<(b)?(a):(b))
#define MAX(a,b) ((a)>(b)?(a):(b))

#ifdef _ENABLE_OPENACC_
#   define OPENACC_ENABLED 1
#   include <openacc.h>
#else
#   define OPENACC_ENABLED 0
#endif

#ifdef _ENABLE_CUDA_
#   define CUDA_ENABLED 1
#else
#   define CUDA_ENABLED 0
#endif

#ifdef _ENABLE_CUDA_KERNEL_
#   define CUDA_KERNEL_ENABLED 1
#else
#   define CUDA_KERNEL_ENABLED 0
#endif

#ifndef BENCHMARK
#   define BENCHMARK "MPI%s BENCHMARK NAME UNSET"
#endif

#ifdef PACKAGE_VERSION
#   define HEADER "# " BENCHMARK " v" PACKAGE_VERSION "\n"
#else
#   define HEADER "# " BENCHMARK "\n"
#endif

#ifndef FIELD_WIDTH
#   define FIELD_WIDTH 20
#endif

#ifndef FLOAT_PRECISION
#   define FLOAT_PRECISION 2
#endif

#define SYNC_MODE (UPC_IN_ALLSYNC | UPC_OUT_ALLSYNC)

#define CHECK(stmt)                                              \
do {                                                             \
   int errno = (stmt);                                           \
   if (0 != errno) {                                             \
       fprintf(stderr, "[%s:%d] function call failed with %d \n",\
        __FILE__, __LINE__, errno);                              \
       exit(EXIT_FAILURE);                                       \
   }                                                             \
   assert(0 == errno);                                           \
} while (0)

#define MPI_CHECK(stmt)                                          \
do {                                                             \
   int mpi_errno = (stmt);                                       \
   if (MPI_SUCCESS != mpi_errno) {                               \
       fprintf(stderr, "[%s:%d] MPI call failed with %d \n",     \
        __FILE__, __LINE__,mpi_errno);                           \
       exit(EXIT_FAILURE);                                       \
   }                                                             \
   assert(MPI_SUCCESS == mpi_errno);                             \
} while (0)

#define TIME() getMicrosecondTimeStamp()
double getMicrosecondTimeStamp();

void print_header_coll (int rank, int full) __attribute__((unused));
void print_header_nbc (int rank, int full);
void print_data (int rank, int full, int size, double avg_time, double
min_time, double max_time, int iterations) __attribute__((unused));
void print_data_nbc (int rank, int full, int size, double ovrl, double
cpu, double comm, double wait, double init, int iterations);

void allocate_host_arrays();

void
calculate_and_print_stats(int rank, int size, int numprocs,
                          double timer, double latency,
                          double test_time, double cpu_time,
                          double wait_time, double init_time);


enum mpi_req{
    MAX_REQ_NUM = 1000
};

#define BW_LOOP_SMALL 100
#define BW_SKIP_SMALL 10
#define BW_LOOP_LARGE 20
#define BW_SKIP_LARGE 2
#define LAT_LOOP_SMALL 10000
#define LAT_SKIP_SMALL 100
#define LAT_LOOP_LARGE 1000
#define LAT_SKIP_LARGE 10
#define COLL_LOOP_SMALL 1000
#define COLL_SKIP_SMALL 100
#define COLL_LOOP_LARGE 100
#define COLL_SKIP_LARGE 10
#define OSHM_LOOP_SMALL 1000
#define OSHM_LOOP_LARGE 100
#define OSHM_SKIP_SMALL 200
#define OSHM_SKIP_LARGE 10
#define OSHM_LOOP_SMALL_MR 500
#define OSHM_LOOP_LARGE_MR 50
#define OSHM_LOOP_ATOMIC 500
#define UPC_LOOP_SMALL 1000
#define UPC_LOOP_LARGE 100
#define UPC_SKIP_SMALL 200
#define UPC_SKIP_LARGE 10

#define MAX_MESSAGE_SIZE (1 << 22)
#define MAX_MSG_SIZE_PT2PT (1<<20)
#define MAX_MSG_SIZE_COLL (1<<20)
#define MIN_MESSAGE_SIZE 1
#define LARGE_MESSAGE_SIZE 8192

#define MAX_ALIGNMENT 65536
#define MAX_MEM_LIMIT (512*1024*1024)
#define MAX_MEM_LOWER_LIMIT (1*1024*1024)
#define WINDOW_SIZE_LARGE 64
#define MYBUFSIZE MAX_MESSAGE_SIZE
#define ONESBUFSIZE ((MAX_MESSAGE_SIZE * WINDOW_SIZE_LARGE) + MAX_ALIGNMENT)
#define MESSAGE_ALIGNMENT 64
#define MESSAGE_ALIGNMENT_MR (1<<12)
#define MYBUFSIZE_MR (MAX_MESSAGE_SIZE * OSHM_LOOP_LARGE_MR + MESSAGE_ALIGNMENT)

enum po_ret_type {
    PO_CUDA_NOT_AVAIL,
    PO_OPENACC_NOT_AVAIL,
    PO_BAD_USAGE,
    PO_HELP_MESSAGE,
    PO_VERSION_MESSAGE,
    PO_OKAY,
};

enum accel_type {
    NONE,
    CUDA,
    OPENACC,
    MANAGED
};

enum target_type {
    CPU,
    GPU,
    BOTH
};

enum benchmark_type {
    COLLECTIVE,
    PT2PT,
    ONE_SIDED,
    MBW_MR,
    OSHM,
    UPC,
    UPCXX
};

enum test_subtype {
    BW,
    LAT,
    LAT_MT,
};

enum test_synctype {
    ALL_SYNC,
    ACTIVE_SYNC
};

enum WINDOW {
    WIN_CREATE=0,
#if MPI_VERSION >= 3
    WIN_ALLOCATE,
    WIN_DYNAMIC
#endif
};

/* Synchronization */
enum SYNC {
    LOCK=0,
    PSCW,
    FENCE,
#if MPI_VERSION >= 3
    FLUSH,
    FLUSH_LOCAL,
    LOCK_ALL,
#endif
};

/*variables*/
extern char const *win_info[20];
extern char const *sync_info[20];

extern MPI_Aint disp_remote;
extern MPI_Aint disp_local;

struct options_t {
    enum accel_type accel;
    enum target_type target;
    int show_size;
    int show_full;
    size_t min_message_size;
    size_t max_message_size;
    size_t iterations;
    size_t iterations_large;
    size_t max_mem_limit;
    size_t skip;
    size_t skip_large;
    size_t window_size_large;
    int num_probes;
    int device_array_size;

    enum benchmark_type bench;
    enum test_subtype  subtype;
    enum test_synctype synctype;

    char src;
    char dst;
    int num_threads;
    char managedSend;
    char managedRecv;
    enum WINDOW win;
    enum SYNC sync;

    int window_size;
    int window_varied;
    int print_rate;
    int pairs;
};


extern struct options_t options;


/*
 * Non-blocking Collectives
 */
double call_test(int * num_tests, MPI_Request** request);
void allocate_device_arrays(int n);
double dummy_compute(double target_secs, MPI_Request *request);
void init_arrays(double seconds);
double do_compute_and_probe(double seconds, MPI_Request *request);
void free_host_arrays();

#ifdef _ENABLE_CUDA_KERNEL_
void free_device_arrays();
#endif

/*
 * Option Processing
 */
int process_options (int argc, char *argv[]);

/*
 * Print Information
 */
void print_bad_usage_message (int rank);
void print_help_message (int rank);
void print_version_message (int rank);
void print_preamble (int rank);
void print_preamble_nbc (int rank);
void print_stats (int rank, int size, double avg, double min, double max);
void print_stats_nbc (int rank, int size, double ovrl, double cpu, double comm,
                      double wait, double init, double test);
/*
 * Memory Management
 */
int allocate_memory_coll (void ** buffer, size_t size, enum accel_type type);
void free_buffer (void * buffer, enum accel_type type);
void set_buffer (void * buffer, enum accel_type type, int data, size_t size);

/*
 * CUDA Context Management
 */
int init_accel (void);
int cleanup_accel (void);

/*
 * Set Benchmark Properties
 */
void set_header (const char * header);
void set_benchmark_name (const char * name);
void enable_accel_support (void);

extern MPI_Request request[MAX_REQ_NUM];
extern MPI_Status  reqstat[MAX_REQ_NUM];
extern MPI_Request send_request[MAX_REQ_NUM];
extern MPI_Request recv_request[MAX_REQ_NUM];

#ifdef _ENABLE_CUDA_
extern CUcontext cuContext;
#endif


#define DEF_NUM_THREADS 2
#define MIN_NUM_THREADS 1
#define MAX_NUM_THREADS 128

#define WINDOW_SIZES {8, 16, 32, 64, 128}
#define WINDOW_SIZES_COUNT   (5)

void usage_mbw_mr();
int allocate_memory_pt2pt (char **sbuf, char **rbuf, int rank);
void print_header_pt2pt (int rank, int type);
void free_memory (void *sbuf, void *rbuf, int rank);
void print_header(int rank, int full);
void allocate_memory_one_sided(int, char *, char *, char **, char **,
                     char **win_base, int, enum WINDOW, MPI_Win *);
void free_memory_one_sided (void *, void *, MPI_Win, int);
void allocate_atomic_memory(int, char *, char *, char *,
                            char *, char **, char **, char **, char **,
                            char **win_base, int, enum WINDOW, MPI_Win *);
void free_atomic_memory (void *, void *, void *, void *, MPI_Win, int);
void usage_one_sided (char const *);
void print_header_one_sided (int, enum WINDOW, enum SYNC);

void print_help_message_get_acc_lat (int);

void print_header_pgas (const char *header, int rank, int full);
void print_data_pgas (int rank, int full, int size, double avg_time, double min_time, double max_time, int iterations);
void print_usage_pgas(int rank, const char * prog, int has_size);
void print_version_pgas(const char *header);
void usage_oshm_pt2pt(int myid);
void wtime(double *t);
