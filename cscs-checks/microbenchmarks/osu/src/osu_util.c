/*
 * Copyright (C) 2002-2017 the Network-Based Computing Laboratory
 * (NBCL), The Ohio State University.
 *
 * Contact: Dr. D. K. Panda (panda@cse.ohio-state.edu)
 *
 * For detailed copyright and licensing information, please refer to the
 * copyright file COPYRIGHT in the top level directory.
 */

#include "osu_util.h"

MPI_Request request[MAX_REQ_NUM];
MPI_Status  reqstat[MAX_REQ_NUM];
MPI_Request send_request[MAX_REQ_NUM];
MPI_Request recv_request[MAX_REQ_NUM];

#ifdef _ENABLE_OPENACC_
#include <openacc.h>
#endif

/*
 * GLOBAL VARIABLES
 */
#ifdef _ENABLE_CUDA_
CUcontext cuContext;
#endif

char const *win_info[20] = {
    "MPI_Win_create",
#if MPI_VERSION >=3
    "MPI_Win_allocate",
    "MPI_Win_create_dynamic",
#endif
};

char const *sync_info[20] = {
    "MPI_Win_lock/unlock",
    "MPI_Win_post/start/complete/wait",
    "MPI_Win_fence",
#if MPI_VERSION >=3
    "MPI_Win_flush",
    "MPI_Win_flush_local",
    "MPI_Win_lock_all/unlock_all",
#endif
};

MPI_Aint disp_remote;
MPI_Aint disp_local;

int mem_on_dev;

static char const * benchmark_header = NULL;
static char const * benchmark_name = NULL;
static int benchmark_type;
static int accel_enabled = 0;
struct options_t options;

/* A is the A in DAXPY for the Compute Kernel */
#define A 2.0
#define DEBUG 0
/*
 * We are using a 2-D matrix to perform dummy
 * computation in non-blocking collective benchmarks
 */
#define DIM 25
static float **a, *x, *y;

#ifdef _ENABLE_CUDA_KERNEL_
/* Using new stream for kernels on gpu */
static cudaStream_t stream;

static int is_alloc = 0;

/* Arrays on device for dummy compute */
static float *d_x, *d_y;
#endif


static struct {
    char const * message;
    char const * optarg;
    int opt;
} bad_usage;


void
print_header(int rank, int full)
{
    switch(options.bench) {
        case PT2PT :
            if (0 == rank) {
                switch (options.accel) {
                    case CUDA:
                        printf(benchmark_header, "-CUDA");
                        break;
                    case OPENACC:
                        printf(benchmark_header, "-OPENACC");
                        break;
                    default:
                        printf(benchmark_header, "");
                        break;
                }

                switch (options.accel) {
                    case CUDA:
                    case OPENACC:
                        fprintf(stdout, "# Send Buffer on %s and Receive Buffer on %s\n",
                               'M' == options.src ? "MANAGED (M)" : ('D' == options.src ? "DEVICE (D)" : "HOST (H)"),
                               'M' == options.dst ? "MANAGED (M)" : ('D' == options.dst ? "DEVICE (D)" : "HOST (H)"));
                    default:
                        if (options.subtype == BW) {
                            fprintf(stdout, "%-*s%*s\n", 10, "# Size", FIELD_WIDTH, "Bandwidth (MB/s)");
                        }
                        else {
                            fprintf(stdout, "%-*s%*s\n", 10, "# Size", FIELD_WIDTH, "Latency (us)");
                        }
                        fflush(stdout);
                }
            }
            break;
        case COLLECTIVE :
            if(rank == 0) {
                fprintf(stdout, HEADER, "");

                if (options.show_size) {
                    fprintf(stdout, "%-*s", 10, "# Size");
                    fprintf(stdout, "%*s", FIELD_WIDTH, "Avg Latency(us)");
                }

                else {
                    fprintf(stdout, "# Avg Latency(us)");
                }

                if (full) {
                    fprintf(stdout, "%*s", FIELD_WIDTH, "Min Latency(us)");
                    fprintf(stdout, "%*s", FIELD_WIDTH, "Max Latency(us)");
                    fprintf(stdout, "%*s\n", 12, "Iterations");
                }

                else {
                    fprintf(stdout, "\n");
                }

                fflush(stdout);
            }
            break;
        default:
            break;
    }
}

void print_header_pgas (const char *header, int rank, int full)
{
    if(rank == 0) {
        fprintf(stdout, header, "");

        if (options.show_size) {
            fprintf(stdout, "%-*s", 10, "# Size");
            fprintf(stdout, "%*s", FIELD_WIDTH, "Avg Latency(us)");
        }

        else {
            fprintf(stdout, "# Avg Latency(us)");
        }

        if (full) {
            fprintf(stdout, "%*s", FIELD_WIDTH, "Min Latency(us)");
            fprintf(stdout, "%*s", FIELD_WIDTH, "Max Latency(us)");
            fprintf(stdout, "%*s\n", 12, "Iterations");
        }

        else {
            fprintf(stdout, "\n");
        }

        fflush(stdout);
    }
}

void print_data_pgas (int rank, int full, int size, double avg_time, double
min_time, double max_time, int iterations)
{
    if(rank == 0) {
        if (size) {
            fprintf(stdout, "%-*d", 10, size);
            fprintf(stdout, "%*.*f", FIELD_WIDTH, FLOAT_PRECISION, avg_time);
        }

        else {
            fprintf(stdout, "%*.*f", 17, FLOAT_PRECISION, avg_time);
        }

        if (full) {
            fprintf(stdout, "%*.*f%*.*f%*d\n",
                    FIELD_WIDTH, FLOAT_PRECISION, min_time,
                    FIELD_WIDTH, FLOAT_PRECISION, max_time,
                    12, iterations);
        }

        else {
            fprintf(stdout, "\n");
        }

        fflush(stdout);
    }
}

void print_header_one_sided (int rank, enum WINDOW win, enum SYNC sync)
{
    if(rank == 0) {
        switch (options.accel) {
            case CUDA:
                printf(benchmark_header, "-CUDA");
                break;
            case OPENACC:
                printf(benchmark_header, "-OPENACC");
                break;
            default:
                printf(benchmark_header, "");
                break;
        }
        fprintf(stdout, "# Window creation: %s\n",
                win_info[win]);
        fprintf(stdout, "# Synchronization: %s\n",
                sync_info[sync]);

        switch (options.accel) {
            case CUDA:
            case OPENACC:
                fprintf(stdout, "# Rank 0 Memory on %s and Rank 1 Memory on %s\n",
                       'D' == options.src ? "DEVICE (D)" : "HOST (H)",
                       'D' == options.dst ? "DEVICE (D)" : "HOST (H)");
            default:
                if (options.subtype == BW) {
                    fprintf(stdout, "%-*s%*s\n", 10, "# Size", FIELD_WIDTH, "Bandwidth (MB/s)");
                } else {
                    fprintf(stdout, "%-*s%*s\n", 10, "# Size", FIELD_WIDTH, "Latency (us)");
                }
                fflush(stdout);
        }
    }
}

void print_data (int rank, int full, int size, double avg_time,
                 double min_time, double max_time, int iterations)
{
    if(rank == 0) {
        if (options.show_size) {
            fprintf(stdout, "%-*d", 10, size);
            fprintf(stdout, "%*.*f", FIELD_WIDTH, FLOAT_PRECISION, avg_time);
        } else {
            fprintf(stdout, "%*.*f", 17, FLOAT_PRECISION, avg_time);
        }

        if (full) {
            fprintf(stdout, "%*.*f%*.*f%*d\n",
                    FIELD_WIDTH, FLOAT_PRECISION, min_time,
                    FIELD_WIDTH, FLOAT_PRECISION, max_time,
                    12, iterations);
        } else {
            fprintf(stdout, "\n");

        }

        fflush(stdout);
    }
}


static int
set_min_message_size (long long value)
{
    if (0 >= value) {
        return -1;
    }

    options.min_message_size = value;

    return 0;
}

static int
set_max_message_size (long long value)
{
    if (0 > value) {
        return -1;
    }

    options.max_message_size = value;

    return 0;
}

static int
set_message_size (char *val_str)
{
    int retval = -1;
    int i, count = 0;
    char *val1, *val2;

    for (i=0; val_str[i]; i++) {
        if (val_str[i] == ':')
            count++;
    }

    if (!count) {
        retval = set_max_message_size(atoll(val_str));
    } else if (count == 1) {
        val1 = strtok(val_str, ":");
        val2 = strtok(NULL, ":");

        if (val1 && val2) {
            retval = set_min_message_size(atoll(val1));
            retval = set_max_message_size(atoll(val2));
        } else if (val1) {
            if (val_str[0] == ':') {
                retval = set_max_message_size(atoll(val1));
            } else {
                retval = set_min_message_size(atoll(val1));
            }
        }
    }

    return retval;
}

static int set_num_warmup (int value)
{
    if (0 > value) {
        return -1;
    }

    options.skip = value;
    options.skip_large = value;

    return 0;
}

static int set_num_iterations (int value)
{
    if (1 > value) {
        return -1;
    }

    options.iterations = value;
    options.iterations_large = value;

    return 0;
}

static int set_window_size_large (int value)
{
    if (1 > value) {
        return -1;
    }

    options.window_size_large = value;

    return 0;
}

static int set_window_size (int value)
{
    if (1 > value) {
        return -1;
    }

    options.window_size = value;

    return 0;
}

static int set_device_array_size (int value)
{
    if (value < 1 ) {
        return -1;
    }

    options.device_array_size = value;

    return 0;
}

void set_device_memory (void * ptr, int data, size_t size)
{
#ifdef _ENABLE_OPENACC_
    size_t i;
    char * p = (char *)ptr;
#endif

    switch (options.accel) {
#ifdef _ENABLE_CUDA_
        case CUDA:
            cudaMemset(ptr, data, size);
            break;
#endif
#ifdef _ENABLE_OPENACC_
        case OPENACC:
#pragma acc parallel copyin(size) deviceptr(p)
            for(i = 0; i < size; i++) {
                p[i] = data;
            }
            break;
#endif
        default:
            break;
    }
}

int free_device_buffer (void * buf)
{
    switch (options.accel) {
#ifdef _ENABLE_CUDA_
        case CUDA:
            cudaFree(buf);
            break;
#endif
#ifdef _ENABLE_OPENACC_
        case OPENACC:
            acc_free(buf);
            break;
#endif
        default:
            /* unknown device */
            return 1;
    }

    return 0;
}

void *align_buffer (void * ptr, unsigned long align_size)
{
    unsigned long buf = (((unsigned long)ptr + (align_size - 1)) / align_size * align_size);
    return (void *) buf;
}


static int set_num_probes (int value)
{
    if (value < 0 ) {
        return -1;
    }

    options.num_probes = value;

    return 0;
}

static int set_max_memlimit (int value)
{
    options.max_mem_limit = value;

    if (value < MAX_MEM_LOWER_LIMIT) {
        options.max_mem_limit = MAX_MEM_LOWER_LIMIT;
        fprintf(stderr,"Requested memory limit too low, using [%d] instead.",
                MAX_MEM_LOWER_LIMIT);
    }

    return 0;
}

void set_header (const char * header)
{
    benchmark_header = header;
}

void set_benchmark_name (const char * name)
{
    benchmark_name = name;
}

void enable_accel_support (void)
{
    accel_enabled = (CUDA_ENABLED || OPENACC_ENABLED);
}

void usage_one_sided (char const * name)
{
    if (accel_enabled) {
        fprintf(stdout, "Usage: %s [options] [RANK0 RANK1] \n", name);
        fprintf(stdout, "RANK0 and RANK1 may be `D' or `H' which specifies whether\n"
                       "the buffer is allocated on the accelerator device or host\n"
                       "memory for each mpi rank\n\n");
    } else {
        fprintf(stdout, "Usage: %s [options] \n", name);
    }

    fprintf(stdout, "Options:\n");

    fprintf(stdout, "  -d --accelerator <type>       accelerator device buffers can be of <type> "
                   "`cuda' or `openacc'\n");
    fprintf(stdout, "\n");

#if MPI_VERSION >= 3
    fprintf(stdout, "  -w --win-option <win_option>\n");
    fprintf(stdout, "            <win_option> can be any of the follows:\n");
    fprintf(stdout, "            create            use MPI_Win_create to create an MPI Window object\n");
    if (accel_enabled) {
        fprintf(stdout, "            allocate          use MPI_Win_allocate to create an MPI Window object (not valid when using device memory)\n");
    } else {
        fprintf(stdout, "            allocate          use MPI_Win_allocate to create an MPI Window object\n");
    }
    fprintf(stdout, "            dynamic           use MPI_Win_create_dynamic to create an MPI Window object\n");
    fprintf(stdout, "\n");
#endif

    fprintf(stdout, "  -s, --sync-option <sync_option>\n");
    fprintf(stdout, "            <sync_option> can be any of the follows:\n");
    fprintf(stdout, "            pscw              use Post/Start/Complete/Wait synchronization calls \n");
    fprintf(stdout, "            fence             use MPI_Win_fence synchronization call\n");
    if (options.synctype == ALL_SYNC) {
        fprintf(stdout, "            lock              use MPI_Win_lock/unlock synchronizations calls\n");
#if MPI_VERSION >= 3
        fprintf(stdout, "            flush             use MPI_Win_flush synchronization call\n");
        fprintf(stdout, "            flush_local       use MPI_Win_flush_local synchronization call\n");
        fprintf(stdout, "            lock_all          use MPI_Win_lock_all/unlock_all synchronization calls\n");
#endif
    }
    fprintf(stdout, "\n");
    if (options.show_size) {
        fprintf(stdout, "  -m, --message-size          [MIN:]MAX  set the minimum and/or the maximum message size to MIN and/or MAX\n");
        fprintf(stdout, "                              bytes respectively. Examples:\n");
        fprintf(stdout, "                              -m 128      // min = default, max = 128\n");
        fprintf(stdout, "                              -m 2:128    // min = 2, max = 128\n");
        fprintf(stdout, "                              -m 2:       // min = 2, max = default\n");
        fprintf(stdout, "  -M, --mem-limit SIZE        set per process maximum memory consumption to SIZE bytes\n");
        fprintf(stdout, "                              (default %d)\n", MAX_MEM_LIMIT);
    }
    fprintf(stdout, "  -x, --warmup ITER           number of warmup iterations to skip before timing"
                   "(default 100)\n");
    fprintf(stdout, "  -i, --iterations ITER       number of iterations for timing (default 10000)\n");

    fprintf(stdout, "  -h, --help                  print this help message\n");
    fflush(stdout);
}

void usage_mbw_mr() {
    fprintf(stdout, "Options:\n");
    fprintf(stdout, "  -R=<0,1>, --print-rate         Print uni-directional message rate (default 1)\n");
    fprintf(stdout, "  -p=<pairs>, --num-pairs        Number of pairs involved (default np / 2)\n");
    fprintf(stdout, "  -W=<window>, --window-size     Number of messages sent before acknowledgement (64, 10)\n");
    fprintf(stdout, "                                 [cannot be used with -v]\n");
    fprintf(stdout, "  -V, --vary-window              Vary the window size (default no)\n");
    fprintf(stdout, "                                 [cannot be used with -w]\n");
    if (options.show_size) {
        fprintf(stdout, "  -m, --message-size          [MIN:]MAX  set the minimum and/or the maximum message size to MIN and/or MAX\n");
        fprintf(stdout, "                              bytes respectively. Examples:\n");
        fprintf(stdout, "                              -m 128      // min = default, max = 128\n");
        fprintf(stdout, "                              -m 2:128    // min = 2, max = 128\n");
        fprintf(stdout, "                              -m 2:       // min = 2, max = default\n");
        fprintf(stdout, "  -M, --mem-limit SIZE        set per process maximum memory consumption to SIZE bytes\n");
        fprintf(stdout, "                              (default %d)\n", MAX_MEM_LIMIT);
    }
    fprintf(stdout, "  -h, --help                     Print this help\n");
    fprintf(stdout, "\n");
    fprintf(stdout, "  Note: This benchmark relies on block ordering of the ranks.  Please see\n");
    fprintf(stdout, "        the README for more information.\n");
    fflush(stdout);
}

void print_usage_pgas(int rank, const char * prog, int has_size)
{
    if (rank == 0) {
        if (has_size) {
            fprintf(stdout, " USAGE : %s [-m SIZE] [-i ITER] [-f] [-hv] [-M SIZE]\n", prog);
            fprintf(stdout, "  -m, --message-size : Set maximum message size to SIZE.\n");
            fprintf(stdout, "                       By default, the value of SIZE is 1MB.\n");
            fprintf(stdout, "  -i, --iterations   : Set number of iterations per message size to ITER.\n");
            fprintf(stdout, "                       By default, the value of ITER is 1000 for small messages\n");
            fprintf(stdout, "                       and 100 for large messages.\n");
            fprintf(stdout, "  -M, --mem-limit    : Set maximum memory consumption (per process) to SIZE. \n");
            fprintf(stdout, "                       By default, the value of SIZE is 512MB.\n");
        }

        else {
            fprintf(stdout, " USAGE : %s [-i ITER] [-f] [-hv] \n", prog);
            fprintf(stdout, "  -i, --iterations   : Set number of iterations to ITER.\n");
            fprintf(stdout, "                       By default, the value of ITER is 1000.\n");
        }

        fprintf(stdout, "  -f, --full         : Print full format listing.  With this option\n");
        fprintf(stdout, "                      the MIN/MAX latency and number of ITERATIONS are\n");
        fprintf(stdout, "                      printed out in addition to the AVERAGE latency.\n");

        fprintf(stdout, "  -h, --help         : Print this help.\n");
        fprintf(stdout, "  -v, --version      : Print version info.\n");
        fprintf(stdout, "\n");
        fflush(stdout);
    }
}

void usage_oshm_pt2pt(int myid)
{
    if(myid == 0) {
        fprintf(stderr, "Invalid arguments. Usage: <prog_name> <heap|global>\n");
    }
}

void print_bad_usage_message (int rank)
{
    if (rank) {
        return;
    }

    if (bad_usage.optarg) {
        fprintf(stderr, "%s [-%c %s]\n\n", bad_usage.message,
                (char)bad_usage.opt, bad_usage.optarg);
    } else {
        fprintf(stderr, "%s [-%c]\n\n", bad_usage.message,
                (char)bad_usage.opt);
    }

    print_help_message(rank);
}

int process_options (int argc, char *argv[])
{
    extern char * optarg;
    extern int optind, optopt;

    char const * optstring = NULL;
    int c;

    int option_index = 0;

    static struct option long_options[] = {
            {"help",            no_argument,        0,  'h'},
            {"version",         no_argument,        0,  'v'},
            {"full",            no_argument,        0,  'f'},
            {"message-size",    required_argument,  0,  'm'},
            {"window-size",     required_argument,  0,  'W'},
            {"num-test-calls",  required_argument,  0,  't'},
            {"iterations",      required_argument,  0,  'i'},
            {"warmup",          required_argument,  0,  'x'},
            {"array-size",      required_argument,  0,  'a'},
            {"sync-option",     required_argument,  0,  's'},
            {"win-options",     required_argument,  0,  'w'},
            {"mem-limit",       required_argument,  0,  'M'},
            {"accelerator",     required_argument,  0,  'd'},
            {"cuda-target",     required_argument,  0,  'r'},
            {"print-rate",      required_argument,  0,  'R'},
            {"num-pairs",       required_argument,  0,  'p'},
            {"vary-window",     required_argument,  0,  'V'}
    };

    enable_accel_support();

    if(options.bench == PT2PT) {
        if (accel_enabled) {
            optstring = (LAT_MT == options.subtype) ? "+:x:i:t:m:hv" : "+:x:i:m:d:hv";
        } else{
            optstring = (LAT_MT == options.subtype) ? "+:hvm:x:i:t:" : "+:hvm:x:i:";
        }
    } else if (options.bench == COLLECTIVE) {
        optstring = "+:hvfm:i:x:M:t:a:";
        if (accel_enabled) {
            optstring = (CUDA_KERNEL_ENABLED) ? "+:d:hvfm:i:x:M:t:r:a:" : "+:d:hvfm:i:x:M:t:a:";
        }
    } else if (options.bench == ONE_SIDED) {
#if MPI_VERSION >= 3
        optstring = (accel_enabled) ? "+:w:s:hvm:d:x:i:" : "+:w:s:hvm:x:i:";
#else
        optstring = (accel_enabled) ? "+:s:hvm:d:x:i:" : "+s:hvm:x:i:";
#endif
    } else if (options.bench == MBW_MR){
        optstring = "p:W:R:x:i:m:Vhv";
    } else if (options.bench == OSHM || options.bench == UPC || options.bench == UPCXX) {
        optstring = ":hvfm:i:M:";
    } else {
        fprintf(stderr,"Invalid benchmark type");
        exit(1);
    }

    /* Set default options*/
    options.accel = NONE;
    options.show_size = 1;
    options.show_full = 0;
    options.num_probes = 0;
    options.device_array_size = 32;
    options.target = CPU;
    options.min_message_size = MIN_MESSAGE_SIZE;
    if (options.bench == COLLECTIVE) {
        options.max_message_size = MAX_MSG_SIZE_COLL;
    } else {
        options.max_message_size = MAX_MESSAGE_SIZE;
    }
    options.max_mem_limit = MAX_MEM_LIMIT;
    options.window_size_large = WINDOW_SIZE_LARGE;
    options.window_size = WINDOW_SIZE_LARGE;
    options.window_varied = 0;
    options.print_rate = 1;

    options.src = 'H';
    options.dst = 'H';

    switch (options.subtype) {
        case BW:
            options.iterations = BW_LOOP_SMALL;
            options.skip = BW_SKIP_SMALL;
            options.iterations_large = BW_LOOP_LARGE;
            options.skip_large = BW_SKIP_LARGE;
            break;
        case LAT_MT:
            options.num_threads = DEF_NUM_THREADS;
            options.min_message_size = 0;
        case LAT:
            if (options.bench == COLLECTIVE) {
                options.iterations = COLL_LOOP_SMALL;
                options.skip = COLL_SKIP_SMALL;
                options.iterations_large = COLL_LOOP_LARGE;
                options.skip_large = COLL_SKIP_LARGE;
            } else {
                options.iterations = LAT_LOOP_SMALL;
                options.skip = LAT_SKIP_SMALL;
                options.iterations_large = LAT_LOOP_LARGE;
                options.skip_large = LAT_SKIP_LARGE;
            }
            if (options.bench == PT2PT) {
                options.min_message_size = 0;
            }
            break;
        default:
            break;
    }

    switch (options.bench) {
        case UPCXX:
        case UPC:
            options.show_size = 0;
        case OSHM:
            options.iterations = OSHM_LOOP_SMALL;
            options.skip = OSHM_SKIP_SMALL;
            options.iterations_large = OSHM_LOOP_LARGE;
            options.skip_large = OSHM_SKIP_LARGE;
            options.max_message_size = 1<<20;
            break;
        default:
            break;
    }

    while ((c = getopt_long(argc, argv, optstring, long_options, &option_index)) != -1) {
        bad_usage.opt = c;
        bad_usage.optarg = NULL;
        bad_usage.message = NULL;

        switch(c) {
            case 'h':
                return PO_HELP_MESSAGE;
            case 'v':
                return PO_VERSION_MESSAGE;
            case 'm':
                if (set_message_size(optarg)) {
                    bad_usage.message = "Invalid Message Size";
                    bad_usage.optarg = optarg;

                    return PO_BAD_USAGE;
                }
                break;
            case 't':
                if (options.bench == COLLECTIVE) {
                    if (set_num_probes(atoi(optarg))){
                        bad_usage.message = "Invalid Number of Probes";
                        bad_usage.optarg = optarg;

                        return PO_BAD_USAGE;
                    }
                } else if (options.bench == PT2PT) {
                    options.num_threads = atoi(optarg);
                    if (options.num_threads < MIN_NUM_THREADS
                        || options.num_threads >= MAX_NUM_THREADS) {
                        bad_usage.message = "Invalid Number of Threads";
                        bad_usage.optarg = optarg;

                        return PO_BAD_USAGE;
                    }
                }
                break;
            case 'i':
                if (set_num_iterations(atoi(optarg))) {
                    bad_usage.message = "Invalid Number of Iterations";
                    bad_usage.optarg = optarg;

                    return PO_BAD_USAGE;
                }
                break;
            case 'x':
                if (set_num_warmup(atoi(optarg))) {
                    bad_usage.message = "Invalid Number of Warmup Iterations";
                    bad_usage.optarg = optarg;

                    return PO_BAD_USAGE;
                }
                break;
            case 'R':
                options.print_rate = atoi(optarg);
                if(0 != options.print_rate && 1 != options.print_rate) {
                    return PO_BAD_USAGE;
                }
                break;
            case 'W':
                if (options.bench == MBW_MR) {
                    if (set_window_size(atoi(optarg))) {
                        bad_usage.message = "Invalid Number of Iterations";
                        bad_usage.optarg = optarg;

                        return PO_BAD_USAGE;
                    }
                }
                else {
                    if (set_window_size_large(atoi(optarg))) {
                        bad_usage.message = "Invalid Number of Iterations";
                        bad_usage.optarg = optarg;

                        return PO_BAD_USAGE;
                    }
                }
                break;
            case 'V':
                options.window_varied = 1;
                break;
            case 'p':
                options.pairs = atoi(optarg);
                break;
            case 'a':
                if (set_device_array_size(atoi(optarg))){
                    bad_usage.message = "Invalid Device Array Size";
                    bad_usage.optarg = optarg;

                    return PO_BAD_USAGE;
                }
                break;
            case 'f':
                options.show_full = 1;
                break;
            case 'M':
                /*
                 * This function does not error but prints a warning message if
                 * the value is too low.
                 */
                set_max_memlimit(atoll(optarg));
                break;
            case 'd':
                if (!accel_enabled) {
                    bad_usage.message = "Benchmark Does Not Support "
                            "Accelerator Transfers";
                    bad_usage.optarg = optarg;
                    return PO_BAD_USAGE;
                } else if (0 == strncasecmp(optarg, "cuda", 10)) {
                    if (CUDA_ENABLED) {
                        options.accel = CUDA;
                    } else {
                        bad_usage.message = "CUDA Support Not Enabled\n"
                                "Please recompile benchmark with CUDA support";
                        bad_usage.optarg = optarg;
                        return PO_BAD_USAGE;
                    }
                } else if (0 == strncasecmp(optarg, "managed", 10)) {
                    if (CUDA_ENABLED) {
                        options.accel = MANAGED;
                    } else {
                        bad_usage.message = "CUDA Managed Memory Support Not Enabled\n"
                                "Please recompile benchmark with CUDA support";
                        bad_usage.optarg = optarg;
                        return PO_BAD_USAGE;
                    }
                } else if (0 == strncasecmp(optarg, "openacc", 10)) {
                    if (OPENACC_ENABLED) {
                        options.accel = OPENACC;
                    } else {
                        bad_usage.message = "OpenACC Support Not Enabled\n"
                                "Please recompile benchmark with OpenACC support";
                        bad_usage.optarg = optarg;
                        return PO_BAD_USAGE;
                    }
                } else {
                    bad_usage.message = "Invalid Accel Type Specified";
                    bad_usage.optarg = optarg;
                    return PO_BAD_USAGE;
                }
                break;
            case 'r':
                if (CUDA_KERNEL_ENABLED) {
                    if (0 == strncasecmp(optarg, "cpu", 10)) {
                        options.target = CPU;
                    } else if (0 == strncasecmp(optarg, "gpu", 10)) {
                        options.target = GPU;
                    } else if (0 == strncasecmp(optarg, "both", 10)) {
                        options.target = BOTH;
                    } else {
                        bad_usage.message = "Please use cpu, gpu, or both";
                        bad_usage.optarg = optarg;
                        return PO_BAD_USAGE;
                    }
                } else {
                    bad_usage.message = "CUDA Kernel Support Not Enabled\n"
                            "Please recompile benchmark with CUDA Kernel support";
                    bad_usage.optarg = optarg;
                    return PO_BAD_USAGE;
                }
                break;

#if MPI_VERSION >= 3
            case 'w':
                if (0 == strcasecmp(optarg, "create")) {
                    options.win = WIN_CREATE;
                } else if (0 == strcasecmp(optarg, "allocate")) {
                    options.win = WIN_ALLOCATE;
                } else if (0 == strcasecmp(optarg, "dynamic")) {
                    options.win = WIN_DYNAMIC;
                } else {
                    return PO_BAD_USAGE;
                }
                break;
#endif
            case 's':
                if (0 == strcasecmp(optarg, "pscw")) {
                    options.sync = PSCW;
                } else if (0 == strcasecmp(optarg, "fence")) {
                    options.sync = FENCE;
                } else if (options.synctype== ALL_SYNC) {
                    if (0 == strcasecmp(optarg, "lock")) {
                        options.sync = LOCK;
                    }
#if MPI_VERSION >= 3
                        else if (0 == strcasecmp(optarg, "flush")) {
                        options.sync = FLUSH;
                    } else if (0 == strcasecmp(optarg, "flush_local")) {
                        options.sync = FLUSH_LOCAL;
                    } else if (0 == strcasecmp(optarg, "lock_all")) {
                        options.sync = LOCK_ALL;
                    }
#endif
                    else {
                        return PO_BAD_USAGE;
                    }
                } else {
                    return PO_BAD_USAGE;
                }
                break;

            case ':':
                bad_usage.message = "Option Missing Required Argument";
                bad_usage.opt = optopt;
                return PO_BAD_USAGE;
            default:
                bad_usage.message = "Invalid Option";
                bad_usage.opt = optopt;
                return PO_BAD_USAGE;
        }

    }

    if (accel_enabled) {

        if ((optind + 2) == argc) {
            options.src = argv[optind][0];
            options.dst = argv[optind + 1][0];

            switch (options.src) {
                case 'D':
                case 'H':
                case 'M':
                    if (options.bench != PT2PT && options.bench != ONE_SIDED) {
                        bad_usage.opt = options.src;
                        bad_usage.message = "This argument is only supported for one-sided and pt2pt benchmarks";
                        return PO_BAD_USAGE;
                    }
                    options.accel = CUDA;
                    break;
                default:
                    return PO_BAD_USAGE;
            }

            switch (options.dst) {
                case 'D':
                case 'H':
                case 'M':
                    if (options.bench != PT2PT && options.bench != ONE_SIDED) {
                        bad_usage.opt = options.dst;
                        bad_usage.message = "This argument is only supported for one-sided and pt2pt benchmarks";
                        return PO_BAD_USAGE;
                    }
                    options.accel = CUDA;
                    break;
                default:
                    return PO_BAD_USAGE;
            }
        } else if (optind != argc) {
            return PO_BAD_USAGE;
        }
    }

    return PO_OKAY;
}


void print_help_message (int rank)
{
    if (rank) {
        return;
    }

    if (accel_enabled && (options.bench == PT2PT)) {
        fprintf(stdout, "Usage: %s [options] [RANK0 RANK1]\n\n", benchmark_name);
        fprintf(stdout, "RANK0 and RANK1 may be `D', `H', or 'M' which specifies whether\n"
                       "the buffer is allocated on the accelerator device memory, host\n"
                       "memory or using CUDA Unified memory respectively for each mpi rank\n\n");
    } else {
        fprintf(stdout, "Usage: %s [options]\n", benchmark_name);
        fprintf(stdout, "Options:\n");
    }

    if (accel_enabled && (options.subtype != LAT_MT)) {
        fprintf(stdout, "  -d, --accelerator  TYPE     use accelerator device buffers, which can be of TYPE `cuda', \n");
        fprintf(stdout, "                              `managed' or `openacc' (uses standard host buffers if not specified)\n");
    }

    if (options.show_size) {
        fprintf(stdout, "  -m, --message-size          [MIN:]MAX  set the minimum and/or the maximum message size to MIN and/or MAX\n");
        fprintf(stdout, "                              bytes respectively. Examples:\n");
        fprintf(stdout, "                              -m 128      // min = default, max = 128\n");
        fprintf(stdout, "                              -m 2:128    // min = 2, max = 128\n");
        fprintf(stdout, "                              -m 2:       // min = 2, max = default\n");
        fprintf(stdout, "  -M, --mem-limit SIZE        set per process maximum memory consumption to SIZE bytes\n");
        fprintf(stdout, "                              (default %d)\n", MAX_MEM_LIMIT);
    }

    fprintf(stdout, "  -i, --iterations ITER       set iterations per message size to ITER (default 1000 for small\n");
    fprintf(stdout, "                              messages, 100 for large messages)\n");
    fprintf(stdout, "  -x, --warmup ITER           set number of warmup iterations to skip before timing (default 200)\n");

    if (options.bench == COLLECTIVE) {
        fprintf(stdout, "  -f, --full                  print full format listing (MIN/MAX latency and ITERATIONS\n");
        fprintf(stdout, "                              displayed in addition to AVERAGE latency)\n");

        fprintf(stdout, "  -t, --num_test_calls CALLS  set the number of MPI_Test() calls during the dummy computation, \n");
        fprintf(stdout, "                              set CALLS to 100, 1000, or any number > 0.\n");

        if (CUDA_KERNEL_ENABLED) {
            fprintf(stdout, "  -r, --cuda-target TARGET    set the compute target for dummy computation\n");
            fprintf(stdout, "                              set TARGET to cpu (default) to execute \n");
            fprintf(stdout, "                              on CPU only, set to gpu for executing kernel \n");
            fprintf(stdout, "                              on the GPU only, and set to both for compute on both.\n");

            fprintf(stdout, "  -a, --array-size SIZE       set the size of arrays to be allocated on device (GPU) \n");
            fprintf(stdout, "                              for dummy compute on device (GPU) (default 32) \n");
        }
    }
    if (LAT_MT == options.subtype) {
        fprintf(stdout, "  -t, --num_threads            number of recv threads to test with (min: %d, "
                       "default: %d, max: %d)\n", MIN_NUM_THREADS, DEF_NUM_THREADS,
               MAX_NUM_THREADS);
    }

    fprintf(stdout, "  -h, --help                  print this help\n");
    fprintf(stdout, "  -v, --version               print version info\n");
    fprintf(stdout, "\n");
    fflush(stdout);
}

void print_help_message_get_acc_lat (int rank)
{
    if (rank) {
        return;
    }

    fprintf(stdout, "Usage: ./osu_get_acc_latency -w <win_option>  -s < sync_option> [-x ITER] [-i ITER]\n");
    if (options.show_size) {
        fprintf(stdout, "  -m, --message-size          [MIN:]MAX  set the minimum and/or the maximum message size to MIN and/or MAX\n");
        fprintf(stdout, "                              bytes respectively. Examples:\n");
        fprintf(stdout, "                              -m 128      // min = default, max = 128\n");
        fprintf(stdout, "                              -m 2:128    // min = 2, max = 128\n");
        fprintf(stdout, "                              -m 2:       // min = 2, max = default\n");
        fprintf(stdout, "  -M, --mem-limit SIZE        set per process maximum memory consumption to SIZE bytes\n");
        fprintf(stdout, "                              (default %d)\n", MAX_MEM_LIMIT);
    }

    fprintf(stdout, "  -x ITER       number of warmup iterations to skip before timing"
            "(default 100)\n");
    fprintf(stdout, "  -i ITER       number of iterations for timing (default 10000)\n");
    fprintf(stdout, "\n");
    fprintf(stdout, "win_option:\n");
    fprintf(stdout, "  create            use MPI_Win_create to create an MPI Window object\n");
    fprintf(stdout, "  allocate          use MPI_Win_allocate to create an MPI Window object\n");
    fprintf(stdout, "  dynamic           use MPI_Win_create_dynamic to create an MPI Window object\n");
    fprintf(stdout, "\n");

    fprintf(stdout, "sync_option:\n");
    fprintf(stdout, "  lock              use MPI_Win_lock/unlock synchronizations calls\n");
    fprintf(stdout, "  flush             use MPI_Win_flush synchronization call\n");
    fprintf(stdout, "  flush_local       use MPI_Win_flush_local synchronization call\n");
    fprintf(stdout, "  lock_all          use MPI_Win_lock_all/unlock_all synchronization calls\n");
    fprintf(stdout, "  pscw              use Post/Start/Complete/Wait synchronization calls \n");
    fprintf(stdout, "  fence             use MPI_Win_fence synchronization call\n");
    fprintf(stdout, "\n");

    fflush(stdout);
}

void print_version_pgas(const char *header)
{
    fprintf(stdout, header, "");
    fflush(stdout);
}

void print_version_message (int rank)
{
    if (rank) {
        return;
    }

    switch (options.accel) {
        case CUDA:
            printf(benchmark_header, "-CUDA");
            break;
        case OPENACC:
            printf(benchmark_header, "-OPENACC");
            break;
        case MANAGED:
            printf(benchmark_header, "-CUDA MANAGED");
            break;
        default:
            printf(benchmark_header, "");
            break;
    }

    fflush(stdout);
}

void print_preamble_nbc (int rank)
{
    if (rank) {
        return;
    }

    fprintf(stdout, "\n");

    switch (options.accel) {
        case CUDA:
            printf(benchmark_header, "-CUDA");
            break;
        case OPENACC:
            printf(benchmark_header, "-OPENACC");
            break;
        case MANAGED:
            printf(benchmark_header, "-MANAGED");
            break;
        default:
            printf(benchmark_header, "");
            break;
    }

    fprintf(stdout, "# Overall = Coll. Init + Compute + MPI_Test + MPI_Wait\n\n");

    if (options.show_size) {
        fprintf(stdout, "%-*s", 10, "# Size");
        fprintf(stdout, "%*s", FIELD_WIDTH, "Overall(us)");
    } else {
        fprintf(stdout, "%*s", FIELD_WIDTH, "Overall(us)");
    }

    if (options.show_full) {
        fprintf(stdout, "%*s", FIELD_WIDTH, "Compute(us)");
        fprintf(stdout, "%*s", FIELD_WIDTH, "Coll. Init(us)");
        fprintf(stdout, "%*s", FIELD_WIDTH, "MPI_Test(us)");
        fprintf(stdout, "%*s", FIELD_WIDTH, "MPI_Wait(us)");
        fprintf(stdout, "%*s", FIELD_WIDTH, "Pure Comm.(us)");
        fprintf(stdout, "%*s\n", FIELD_WIDTH, "Overlap(%)");

    } else {
        fprintf(stdout, "%*s", FIELD_WIDTH, "Compute(us)");
        fprintf(stdout, "%*s", FIELD_WIDTH, "Pure Comm.(us)");
        fprintf(stdout, "%*s\n", FIELD_WIDTH, "Overlap(%)");
    }

    fflush(stdout);
}

void display_nbc_params()
{
    if (options.show_full) {
        fprintf(stdout, "%*s", FIELD_WIDTH, "Compute(us)");
        fprintf(stdout, "%*s", FIELD_WIDTH, "Coll. Init(us)");
        fprintf(stdout, "%*s", FIELD_WIDTH, "MPI_Test(us)");
        fprintf(stdout, "%*s", FIELD_WIDTH, "MPI_Wait(us)");
        fprintf(stdout, "%*s", FIELD_WIDTH, "Pure Comm.(us)");
        fprintf(stdout, "%*s\n", FIELD_WIDTH, "Overlap(%)");

    } else {
        fprintf(stdout, "%*s", FIELD_WIDTH, "Compute(us)");
        fprintf(stdout, "%*s", FIELD_WIDTH, "Pure Comm.(us)");
        fprintf(stdout, "%*s\n", FIELD_WIDTH, "Overlap(%)");
    }
}

void print_preamble (int rank)
{
    if (rank) {
        return;
    }

    fprintf(stdout, "\n");

    switch (options.accel) {
        case CUDA:
            printf(benchmark_header, "-CUDA");
            break;
        case OPENACC:
            printf(benchmark_header, "-OPENACC");
            break;
        default:
            printf(benchmark_header, "");
            break;
    }

    if (options.show_size) {
        fprintf(stdout, "%-*s", 10, "# Size");
        fprintf(stdout, "%*s", FIELD_WIDTH, "Avg Latency(us)");
    } else {
        fprintf(stdout, "# Avg Latency(us)");
    }

    if (options.show_full) {
        fprintf(stdout, "%*s", FIELD_WIDTH, "Min Latency(us)");
        fprintf(stdout, "%*s", FIELD_WIDTH, "Max Latency(us)");
        fprintf(stdout, "%*s\n", 12, "Iterations");
    } else {
        fprintf(stdout, "\n");
    }

    fflush(stdout);
}

void calculate_and_print_stats(int rank, int size, int numprocs,
                          double timer, double latency,
                          double test_time, double cpu_time,
                          double wait_time, double init_time)
{
    double test_total   = (test_time * 1e6) / options.iterations;
    double tcomp_total  = (cpu_time * 1e6) / options.iterations;
    double overall_time = (timer * 1e6) / options.iterations;
    double wait_total   = (wait_time * 1e6) / options.iterations;
    double init_total   = (init_time * 1e6) / options.iterations;
    double comm_time   = latency;

    if(rank != 0) {
        MPI_CHECK(MPI_Reduce(&test_total, &test_total, 1, MPI_DOUBLE, MPI_SUM, 0,
                   MPI_COMM_WORLD));
        MPI_CHECK(MPI_Reduce(&comm_time, &comm_time, 1, MPI_DOUBLE, MPI_SUM, 0,
                   MPI_COMM_WORLD));
        MPI_CHECK(MPI_Reduce(&overall_time, &overall_time, 1, MPI_DOUBLE, MPI_SUM, 0,
                   MPI_COMM_WORLD));
        MPI_CHECK(MPI_Reduce(&tcomp_total, &tcomp_total, 1, MPI_DOUBLE, MPI_SUM, 0,
                   MPI_COMM_WORLD));
        MPI_CHECK(MPI_Reduce(&wait_total, &wait_total, 1, MPI_DOUBLE, MPI_SUM, 0,
                   MPI_COMM_WORLD));
        MPI_CHECK(MPI_Reduce(&init_total, &init_total, 1, MPI_DOUBLE, MPI_SUM, 0,
                   MPI_COMM_WORLD));
    } else {
        MPI_CHECK(MPI_Reduce(MPI_IN_PLACE, &test_total, 1, MPI_DOUBLE, MPI_SUM, 0,
                   MPI_COMM_WORLD));
        MPI_CHECK(MPI_Reduce(MPI_IN_PLACE, &comm_time, 1, MPI_DOUBLE, MPI_SUM, 0,
                   MPI_COMM_WORLD));
        MPI_CHECK(MPI_Reduce(MPI_IN_PLACE, &overall_time, 1, MPI_DOUBLE, MPI_SUM, 0,
                   MPI_COMM_WORLD));
        MPI_CHECK(MPI_Reduce(MPI_IN_PLACE, &tcomp_total, 1, MPI_DOUBLE, MPI_SUM, 0,
                   MPI_COMM_WORLD));
        MPI_CHECK(MPI_Reduce(MPI_IN_PLACE, &wait_total, 1, MPI_DOUBLE, MPI_SUM, 0,
                   MPI_COMM_WORLD));
        MPI_CHECK(MPI_Reduce(MPI_IN_PLACE, &init_total, 1, MPI_DOUBLE, MPI_SUM, 0,
                   MPI_COMM_WORLD));
    }

    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    /* Overall Time (Overlapped) */
    overall_time = overall_time/numprocs;
    /* Computation Time */
    tcomp_total = tcomp_total/numprocs;
    /* Time taken by MPI_Test calls */
    test_total = test_total/numprocs;
    /* Pure Communication Time */
    comm_time = comm_time/numprocs;
    /* Time for MPI_Wait() call */
    wait_total = wait_total/numprocs;
    /* Time for the NBC call */
    init_total = init_total/numprocs;

    print_stats_nbc(rank, size, overall_time, tcomp_total, comm_time,
                    wait_total, init_total, test_total);

}

void print_stats_nbc (int rank, int size, double overall_time,
                 double cpu_time, double comm_time,
                 double wait_time, double init_time,
                 double test_time)
{
    if (rank) {
        return;
    }

    double overlap;

    /* Note : cpu_time received in this function includes time for
       *      dummy compute as well as test calls so we will subtract
       *      the test_time for overlap calculation as test is an
       *      overhead
       */

    overlap = MAX(0, 100 - (((overall_time - (cpu_time - test_time)) / comm_time) * 100));

    if (options.show_size) {
        fprintf(stdout, "%-*d", 10, size);
        fprintf(stdout, "%*.*f", FIELD_WIDTH, FLOAT_PRECISION, overall_time);
    } else {
        fprintf(stdout, "%*.*f", FIELD_WIDTH, FLOAT_PRECISION, overall_time);
    }

    if (options.show_full) {
        fprintf(stdout, "%*.*f%*.*f%*.*f%*.*f%*.*f%*.*f\n",
                FIELD_WIDTH, FLOAT_PRECISION, (cpu_time - test_time),
                FIELD_WIDTH, FLOAT_PRECISION, init_time,
                FIELD_WIDTH, FLOAT_PRECISION, test_time,
                FIELD_WIDTH, FLOAT_PRECISION, wait_time,
                FIELD_WIDTH, FLOAT_PRECISION, comm_time,
                FIELD_WIDTH, FLOAT_PRECISION, overlap);
    } else {
        fprintf(stdout, "%*.*f", FIELD_WIDTH, FLOAT_PRECISION, (cpu_time - test_time));
        fprintf(stdout, "%*.*f", FIELD_WIDTH, FLOAT_PRECISION, comm_time);
        fprintf(stdout, "%*.*f\n", FIELD_WIDTH, FLOAT_PRECISION, overlap);
    }

    fflush(stdout);
}

void print_stats (int rank, int size, double avg_time, double min_time, double max_time)
{
    if (rank) {
        return;
    }

    if (options.show_size) {
        fprintf(stdout, "%-*d", 10, size);
        fprintf(stdout, "%*.*f", FIELD_WIDTH, FLOAT_PRECISION, avg_time);
    } else {
        fprintf(stdout, "%*.*f", 17, FLOAT_PRECISION, avg_time);
    }

    if (options.show_full) {
        fprintf(stdout, "%*.*f%*.*f%*lu\n",
                FIELD_WIDTH, FLOAT_PRECISION, min_time,
                FIELD_WIDTH, FLOAT_PRECISION, max_time,
                12, options.iterations);
    } else {
        fprintf(stdout, "\n");
    }

    fflush(stdout);
}

double getMicrosecondTimeStamp()
{
    double retval;
    struct timeval tv;
    if (gettimeofday(&tv, NULL)) {
        perror("gettimeofday");
        abort();
    }
    retval = ((double)tv.tv_sec) * 1000000 + tv.tv_usec;
    return retval;
}

void set_buffer (void * buffer, enum accel_type type, int data, size_t size)
{
#ifdef _ENABLE_OPENACC_
    size_t i;
    char * p = (char *)buffer;
#endif
    switch (type) {
        case NONE:
            memset(buffer, data, size);
            break;
        case CUDA:
        case MANAGED:
#ifdef _ENABLE_CUDA_
            cudaMemset(buffer, data, size);
#endif
            break;
        case OPENACC:
#ifdef _ENABLE_OPENACC_
            #pragma acc parallel loop deviceptr(p)
            for (i = 0; i < size; i++) {
                p[i] = data;
            }
#endif
            break;
    }
}

int allocate_memory_coll (void ** buffer, size_t size, enum accel_type type)
{
    if (options.target == CPU || options.target == BOTH) {
        allocate_host_arrays();
    }

    size_t alignment = sysconf(_SC_PAGESIZE);
#ifdef _ENABLE_CUDA_
    cudaError_t cuerr = cudaSuccess;
#endif

    switch (type) {
        case NONE:
            return posix_memalign(buffer, alignment, size);
#ifdef _ENABLE_CUDA_
        case CUDA:
            cuerr = cudaMalloc(buffer, size);
            if (cudaSuccess != cuerr) {
                return 1;
            } else {
                return 0;
            }
        case MANAGED:
            cuerr = cudaMallocManaged(buffer, size, cudaMemAttachGlobal);
            if (cudaSuccess != cuerr) {
                return 1;
            } else {
                return 0;
            }
#endif
#ifdef _ENABLE_OPENACC_
        case OPENACC:
            *buffer = acc_malloc(size);
            if (NULL == *buffer) {
                return 1;
            } else {
                return 0;
            }
#endif
        default:
            return 1;
    }
}

int allocate_device_buffer (char ** buffer)
{
#ifdef _ENABLE_CUDA_
    cudaError_t cuerr = cudaSuccess;
#endif

    switch (options.accel) {
#ifdef _ENABLE_CUDA_
        case CUDA:
            cuerr = cudaMalloc((void **)buffer, options.max_message_size);

            if (cudaSuccess != cuerr) {
                fprintf(stderr, "Could not allocate device memory\n");
                return 1;
            }
            break;
#endif
#ifdef _ENABLE_OPENACC_
        case OPENACC:
            *buffer = acc_malloc(options.max_message_size);
            if (NULL == *buffer) {
                fprintf(stderr, "Could not allocate device memory\n");
                return 1;
            }
            break;
#endif
        default:
            fprintf(stderr, "Could not allocate device memory\n");
            return 1;
    }

    return 0;
}

int allocate_managed_buffer (char ** buffer)
{
#ifdef _ENABLE_CUDA_
    cudaError_t cuerr = cudaSuccess;
#endif

    switch (options.accel) {
#ifdef _ENABLE_CUDA_
        case CUDA:
            cuerr = cudaMallocManaged((void **)buffer, options.max_message_size, cudaMemAttachGlobal);

            if (cudaSuccess != cuerr) {
                fprintf(stderr, "Could not allocate device memory\n");
                return 1;
            }
            break;
#endif
        default:
            fprintf(stderr, "Could not allocate device memory\n");
            return 1;

    }
    return 0;
}

int allocate_memory_pt2pt (char ** sbuf, char ** rbuf, int rank)
{
    unsigned long align_size = sysconf(_SC_PAGESIZE);

    switch (rank) {
        case 0:
            if ('D' == options.src) {
                if (allocate_device_buffer(sbuf)) {
                    fprintf(stderr, "Error allocating cuda memory\n");
                    return 1;
                }

                if (allocate_device_buffer(rbuf)) {
                    fprintf(stderr, "Error allocating cuda memory\n");
                    return 1;
                }
            } else if ('M' == options.src) {
                if (allocate_managed_buffer(sbuf)) {
                    fprintf(stderr, "Error allocating cuda unified memory\n");
                    return 1;
                }

                if (allocate_managed_buffer(rbuf)) {
                    fprintf(stderr, "Error allocating cuda unified memory\n");
                    return 1;
                }
            } else {
                if (posix_memalign((void**)sbuf, align_size, options.max_message_size)) {
                    fprintf(stderr, "Error allocating host memory\n");
                    return 1;
                }

                if (posix_memalign((void**)rbuf, align_size, options.max_message_size)) {
                    fprintf(stderr, "Error allocating host memory\n");
                    return 1;
                }
            }
            break;
        case 1:
            if ('D' == options.dst) {
                if (allocate_device_buffer(sbuf)) {
                    fprintf(stderr, "Error allocating cuda memory\n");
                    return 1;
                }

                if (allocate_device_buffer(rbuf)) {
                    fprintf(stderr, "Error allocating cuda memory\n");
                    return 1;
                }
            } else if ('M' == options.dst) {
                if (allocate_managed_buffer(sbuf)) {
                    fprintf(stderr, "Error allocating cuda unified memory\n");
                    return 1;
                }

                if (allocate_managed_buffer(rbuf)) {
                    fprintf(stderr, "Error allocating cuda unified memory\n");
                    return 1;
                }
            } else {
                if (posix_memalign((void**)sbuf, align_size, options.max_message_size)) {
                    fprintf(stderr, "Error allocating host memory\n");
                    return 1;
                }

                if (posix_memalign((void**)rbuf, align_size, options.max_message_size)) {
                    fprintf(stderr, "Error allocating host memory\n");
                    return 1;
                }
            }
            break;
    }

    return 0;
}

void allocate_memory_one_sided(int rank, char *sbuf_orig, char *rbuf_orig, char **sbuf, char **rbuf,
                char **win_base, int size, enum WINDOW type, MPI_Win *win)
{
    int page_size;

    page_size = getpagesize();
    assert(page_size <= MAX_ALIGNMENT);

    if (rank == 0) {
        mem_on_dev = ('D' == options.src) ? 1 : 0;
    } else {
        mem_on_dev = ('D' == options.dst) ? 1 : 0;
    }

    if (mem_on_dev) {
        CHECK(allocate_device_buffer(sbuf));
        set_device_memory(*sbuf, 'a', size);
        CHECK(allocate_device_buffer(rbuf));
        set_device_memory(*rbuf, 'b', size);
    } else {
        *sbuf = (char *)align_buffer((void *)sbuf_orig, page_size);
        memset(*sbuf, 'a', size);
        *rbuf = (char *)align_buffer((void *)rbuf_orig, page_size);
        memset(*rbuf, 'b', size);
    }

#if MPI_VERSION >= 3
    MPI_Status  reqstat;

    switch (type) {
        case WIN_CREATE:
            MPI_CHECK(MPI_Win_create(*win_base, size, 1, MPI_INFO_NULL, MPI_COMM_WORLD, win));
            break;
        case WIN_DYNAMIC:
            MPI_CHECK(MPI_Win_create_dynamic(MPI_INFO_NULL, MPI_COMM_WORLD, win));
            MPI_CHECK(MPI_Win_attach(*win, (void *)*win_base, size));
            MPI_CHECK(MPI_Get_address(*win_base, &disp_local));
            if(rank == 0){
                MPI_CHECK(MPI_Send(&disp_local, 1, MPI_AINT, 1, 1, MPI_COMM_WORLD));
                MPI_CHECK(MPI_Recv(&disp_remote, 1, MPI_AINT, 1, 1, MPI_COMM_WORLD, &reqstat));
            } else {
                MPI_CHECK(MPI_Recv(&disp_remote, 1, MPI_AINT, 0, 1, MPI_COMM_WORLD, &reqstat));
                MPI_CHECK(MPI_Send(&disp_local, 1, MPI_AINT, 0, 1, MPI_COMM_WORLD));
            }
            break;
        default:
            if (mem_on_dev) {
                MPI_CHECK(MPI_Win_create(*win_base, size, 1, MPI_INFO_NULL, MPI_COMM_WORLD, win));
            } else {
                MPI_CHECK(MPI_Win_allocate(size, 1, MPI_INFO_NULL, MPI_COMM_WORLD, *win_base, win));
            }
            break;
    }
#else
    MPI_CHECK(MPI_Win_create(*win_base, size, 1, MPI_INFO_NULL, MPI_COMM_WORLD, win));
#endif
}

void free_buffer (void * buffer, enum accel_type type)
{
    switch (type) {
        case NONE:
            free(buffer);
            break;
        case MANAGED:
        case CUDA:
#ifdef _ENABLE_CUDA_
            cudaFree(buffer);
#endif
            break;
        case OPENACC:
#ifdef _ENABLE_OPENACC_
            acc_free(buffer);
#endif
            break;
    }

    /* Free dummy compute related resources */
    if (CPU == options.target || BOTH == options.target) {
        free_host_arrays();
    }

    if (GPU == options.target || BOTH == options.target) {
#ifdef _ENABLE_CUDA_KERNEL_
        free_device_arrays();
#endif /* #ifdef _ENABLE_CUDA_KERNEL_ */
    }
}

int init_accel (void)
{
#if defined(_ENABLE_OPENACC_) || defined(_ENABLE_CUDA_)
    char * str;
     int local_rank, dev_count;
     int dev_id = 0;
#endif
#ifdef _ENABLE_CUDA_
    CUresult curesult = CUDA_SUCCESS;
    CUdevice cuDevice;
#endif

    switch (options.accel) {
#ifdef _ENABLE_CUDA_
        case MANAGED:
        case CUDA:
            if ((str = getenv("LOCAL_RANK")) != NULL) {
                cudaGetDeviceCount(&dev_count);
                local_rank = atoi(str);
                dev_id = local_rank % dev_count;
            }

            curesult = cuInit(0);
            if (curesult != CUDA_SUCCESS) {
                return 1;
            }

            curesult = cuDeviceGet(&cuDevice, dev_id);
            if (curesult != CUDA_SUCCESS) {
                return 1;
            }

            curesult = cuCtxCreate(&cuContext, 0, cuDevice);
            if (curesult != CUDA_SUCCESS) {
                return 1;
            }
            break;
#endif
#ifdef _ENABLE_OPENACC_
        case OPENACC:
            if ((str = getenv("LOCAL_RANK")) != NULL) {
                dev_count = acc_get_num_devices(acc_device_not_host);
                local_rank = atoi(str);
                dev_id = local_rank % dev_count;
            }

            acc_set_device_num (dev_id, acc_device_not_host);
            break;
#endif
        default:
            fprintf(stderr, "Invalid device type, should be cuda or openacc\n");
            return 1;
    }

    return 0;
}

int cleanup_accel (void)
{
#ifdef _ENABLE_CUDA_
    CUresult curesult = CUDA_SUCCESS;
#endif

    switch (options.accel) {
#ifdef _ENABLE_CUDA_
        case MANAGED:
        case CUDA:
            curesult = cuCtxDestroy(cuContext);

            if (curesult != CUDA_SUCCESS) {
                return 1;
            }
            break;
#endif
#ifdef _ENABLE_OPENACC_
        case OPENACC:
            acc_shutdown(acc_device_nvidia);
            break;
#endif
        default:
            fprintf(stderr, "Invalid accel type, should be cuda or openacc\n");
            return 1;
    }

    return 0;
}

#ifdef _ENABLE_CUDA_KERNEL_
void free_device_arrays()
{
    cudaError_t cuerr = cudaSuccess;
    if (is_alloc) {
        cuerr = cudaFree(d_x);
        if (cuerr != cudaSuccess) {
            fprintf(stderr, "Failed to free device array\n");
        }

        cuerr = cudaFree(d_y);
        if (cuerr != cudaSuccess) {
            fprintf(stderr, "Failed to free device array\n");
        }

        is_alloc = 0;
    }
}
#endif

void free_host_arrays()
{
    int i = 0;

    if (x) {
        free(x);
    }
    if (y) {
        free(y);
    }

    if (a) {
        for (i = 0; i < DIM; i++) {
            free(a[i]);
        }
        free(a);
    }

    x = NULL;
    y = NULL;
    a = NULL;
}

void free_memory (void * sbuf, void * rbuf, int rank)
{
    switch (rank) {
        case 0:
            if ('D' == options.src || 'M' == options.src) {
                free_device_buffer(sbuf);
                free_device_buffer(rbuf);
            } else {
                free(sbuf);
                free(rbuf);
            }
            break;
        case 1:
            if ('D' == options.dst || 'M' == options.dst) {
                free_device_buffer(sbuf);
                free_device_buffer(rbuf);
            } else {
                free(sbuf);
                free(rbuf);
            }
            break;
    }
}

void free_memory_one_sided (void *sbuf, void *rbuf, MPI_Win win, int rank)
{
    MPI_CHECK(MPI_Win_free(&win));

    switch (rank) {
        case 0:
            if ('D' == options.src) {
                free_device_buffer(sbuf);
                free_device_buffer(rbuf);
            }
            break;
        case 1:
            if ('D' == options.dst) {
                free_device_buffer(sbuf);
                free_device_buffer(rbuf);
            }
            break;
    }
}

double dummy_compute(double seconds, MPI_Request* request)
{
    double test_time = 0.0;

    test_time = do_compute_and_probe(seconds, request);

    return test_time;
}

#ifdef _ENABLE_CUDA_KERNEL_
void do_compute_gpu(double seconds)
{
    int i,j;
    double time_elapsed = 0.0, t1 = 0.0, t2 = 0.0;

    {
        t1 = MPI_Wtime();

        /* Execute Dummy Kernel on GPU if set by user */
        if (options.target == BOTH || options.target == GPU) {
            {
                cudaStreamCreate(&stream);
                call_kernel(A, d_x, d_y, options.device_array_size, &stream);
            }
        }

        t2 = MPI_Wtime();
        time_elapsed += (t2-t1);
    }
}
#endif

void
compute_on_host()
{
    int i = 0, j = 0;
    for (i = 0; i < DIM; i++)
        for (j = 0; j < DIM; j++)
            x[i] = x[i] + a[i][j]*a[j][i] + y[j];
}


static inline void do_compute_cpu(double target_seconds)
{
    double t1 = 0.0, t2 = 0.0;
    double time_elapsed = 0.0;
    while (time_elapsed < target_seconds) {
        t1 = MPI_Wtime();
        compute_on_host();
        t2 = MPI_Wtime();
        time_elapsed += (t2-t1);
    }
    if (DEBUG) {
        fprintf(stderr, "time elapsed = %f\n", (time_elapsed * 1e6));
    }
}

void wtime(double *t)
{
    static int sec = -1;
    struct timeval tv;
    //gettimeofday(&tv, (void *)0);
    gettimeofday(&tv, 0);
    if (sec < 0) sec = tv.tv_sec;
    *t = (tv.tv_sec - sec)*1.0e+6 + tv.tv_usec;
}

double do_compute_and_probe(double seconds, MPI_Request* request)
{
    double t1 = 0.0, t2 = 0.0;
    double test_time = 0.0;
    int num_tests = 0;
    double target_seconds_for_compute = 0.0;
    int flag = 0;
    MPI_Status status;

    if (options.num_probes) {
        target_seconds_for_compute = (double) seconds/options.num_probes;
        if (DEBUG) {
            fprintf(stderr, "setting target seconds to %f\n", (target_seconds_for_compute * 1e6 ));
        }
    } else {
        target_seconds_for_compute = seconds;
        if (DEBUG) {
            fprintf(stderr, "setting target seconds to %f\n", (target_seconds_for_compute * 1e6 ));
        }
    }

#ifdef _ENABLE_CUDA_KERNEL_
    if (options.target == GPU) {
        if (options.num_probes) {
            /* Do the dummy compute on GPU only */
            do_compute_gpu(target_seconds_for_compute);
            num_tests = 0;
            while (num_tests < options.num_probes) {
                t1 = MPI_Wtime();
                MPI_CHECK(MPI_Test(request, &flag, &status));
                t2 = MPI_Wtime();
                test_time += (t2-t1);
                num_tests++;
            }
        } else {
            do_compute_gpu(target_seconds_for_compute);
        }
    } else if (options.target == BOTH) {
        if (options.num_probes) {
            /* Do the dummy compute on GPU and CPU*/
            do_compute_gpu(target_seconds_for_compute);
            num_tests = 0;
            while (num_tests < options.num_probes) {
                t1 = MPI_Wtime();
                MPI_CHECK(MPI_Test(request, &flag, &status));
                t2 = MPI_Wtime();
                test_time += (t2-t1);
                num_tests++;
                do_compute_cpu(target_seconds_for_compute);
            }
        } else {
            do_compute_gpu(target_seconds_for_compute);
            do_compute_cpu(target_seconds_for_compute);
        }
    } else
#endif
    if (options.target == CPU) {
        if (options.num_probes) {
            num_tests = 0;
            while (num_tests < options.num_probes) {
                do_compute_cpu(target_seconds_for_compute);
                t1 = MPI_Wtime();
                MPI_CHECK(MPI_Test(request, &flag, &status));
                t2 = MPI_Wtime();
                test_time += (t2-t1);
                num_tests++;
            }
        } else {
            do_compute_cpu(target_seconds_for_compute);
        }
    }

#ifdef _ENABLE_CUDA_KERNEL_
    if (options.target == GPU || options.target == BOTH) {
        cudaDeviceSynchronize();
        cudaStreamDestroy(stream);
    }
#endif

    return test_time;
}

void allocate_host_arrays()
{
    int i=0, j=0;
    a = (float **)malloc(DIM * sizeof(float *));

    for (i = 0; i < DIM; i++) {
        a[i] = (float *)malloc(DIM * sizeof(float));
    }

    x = (float *)malloc(DIM * sizeof(float));
    y = (float *)malloc(DIM * sizeof(float));

    for (i = 0; i < DIM; i++) {
        x[i] = y[i] = 1.0f;
        for (j = 0; j < DIM; j++) {
            a[i][j] = 2.0f;
        }
    }
}

void allocate_atomic_memory(int rank, char *sbuf_orig, char *rbuf_orig, char *tbuf_orig,
                       char *cbuf_orig, char **sbuf, char **rbuf, char **tbuf,
                       char **cbuf, char **win_base, int size, enum WINDOW type, MPI_Win *win)
{
    int page_size;

    page_size = getpagesize();
    assert(page_size <= MAX_ALIGNMENT);

    if (rank == 0) {
        mem_on_dev = ('D' == options.src) ? 1 : 0;
    } else {
        mem_on_dev = ('D' == options.dst) ? 1 : 0;
    }

    if (mem_on_dev) {
        CHECK(allocate_device_buffer(sbuf));
        set_device_memory(*sbuf, 'a', size);
        CHECK(allocate_device_buffer(rbuf));
        set_device_memory(*rbuf, 'b', size);
        CHECK(allocate_device_buffer(tbuf));
        set_device_memory(*tbuf, 'c', size);
        if (cbuf != NULL) {
            CHECK(allocate_device_buffer(cbuf));
            set_device_memory(*cbuf, 'a', size);
        }
    } else {
        *sbuf = (char *)align_buffer((void *)sbuf_orig, page_size);
        memset(*sbuf, 'a', size);
        *rbuf = (char *)align_buffer((void *)rbuf_orig, page_size);
        memset(*rbuf, 'b', size);
        *tbuf = (char *)align_buffer((void *)tbuf_orig, page_size);
        memset(*tbuf, 'c', size);
        if (cbuf != NULL) {
            *cbuf = (char *)align_buffer((void *)cbuf_orig, page_size);
            memset(*cbuf, 'a', size);
        }
    }

#if MPI_VERSION >= 3
    MPI_Status  reqstat;

    switch (type) {
        case WIN_CREATE:
            MPI_CHECK(MPI_Win_create(*win_base, size, 1, MPI_INFO_NULL, MPI_COMM_WORLD, win));
            break;
        case WIN_DYNAMIC:
            MPI_CHECK(MPI_Win_create_dynamic(MPI_INFO_NULL, MPI_COMM_WORLD, win));
            MPI_CHECK(MPI_Win_attach(*win, (void *)*win_base, size));
            MPI_CHECK(MPI_Get_address(*win_base, &disp_local));
            if(rank == 0){
                MPI_CHECK(MPI_Send(&disp_local, 1, MPI_AINT, 1, 1, MPI_COMM_WORLD));
                MPI_CHECK(MPI_Recv(&disp_remote, 1, MPI_AINT, 1, 1, MPI_COMM_WORLD, &reqstat));
            } else {
                MPI_CHECK(MPI_Recv(&disp_remote, 1, MPI_AINT, 0, 1, MPI_COMM_WORLD, &reqstat));
                MPI_CHECK(MPI_Send(&disp_local, 1, MPI_AINT, 0, 1, MPI_COMM_WORLD));
            }
            break;
        default:
            if (mem_on_dev) {
                MPI_CHECK(MPI_Win_create(*win_base, size, 1, MPI_INFO_NULL, MPI_COMM_WORLD, win));
            } else {
                MPI_CHECK(MPI_Win_allocate(size, 1, MPI_INFO_NULL, MPI_COMM_WORLD, *win_base, win));
            }
            break;
    }
#else
    MPI_CHECK(MPI_Win_create(*win_base, size, 1, MPI_INFO_NULL, MPI_COMM_WORLD, win));
#endif
}

void free_atomic_memory (void *sbuf, void *rbuf, void *tbuf, void *cbuf, MPI_Win win, int rank)
{
    MPI_CHECK(MPI_Win_free(&win));

    switch (rank) {
        case 0:
            if ('D' == options.src) {
                free_device_buffer(sbuf);
                free_device_buffer(rbuf);
                free_device_buffer(tbuf);
                if (cbuf != NULL)
                    free_device_buffer(cbuf);
            }
            break;
        case 1:
            if ('D' == options.dst) {
                free_device_buffer(sbuf);
                free_device_buffer(rbuf);
                free_device_buffer(tbuf);
                if (cbuf != NULL)
                    free_device_buffer(cbuf);
            }
            break;
    }
}

void init_arrays(double target_time)
{

    if (DEBUG) {
        fprintf(stderr, "called init_arrays with target_time = %f \n",
                (target_time * 1e6));
    }

#ifdef _ENABLE_CUDA_KERNEL_
    if (options.target == GPU || options.target == BOTH) {
    /* Setting size of arrays for Dummy Compute */
    int N = options.device_array_size;

    /* Device Arrays for Dummy Compute */
    allocate_device_arrays(N);

    double time_elapsed = 0.0;
    double t1 = 0.0, t2 = 0.0;

    while (1) {
        t1 = MPI_Wtime();

        if (options.target == GPU || options.target == BOTH) {
            cudaStreamCreate(&stream);
            call_kernel(A, d_x, d_y, N, &stream);

            cudaDeviceSynchronize();
            cudaStreamDestroy(stream);
        }

        t2 = MPI_Wtime();
        if ((t2-t1) < target_time)
        {
            N += 32;

            /* Now allocate arrays of size N */
            allocate_device_arrays(N);
        } else {
            break;
        }
    }

    /* we reach here with desired N so save it and pass it to options */
    options.device_array_size = N;
    if (DEBUG) {
        fprintf(stderr, "correct N = %d\n", N);
        }
    }
#endif

}

#ifdef _ENABLE_CUDA_KERNEL_
void allocate_device_arrays(int n)
{
    cudaError_t cuerr = cudaSuccess;

    /* First free the old arrays */
    free_device_arrays();

    /* Allocate Device Arrays for Dummy Compute */
    cuerr = cudaMalloc((void**)&d_x, n * sizeof(float));
    if (cuerr != cudaSuccess) {
        fprintf(stderr, "Failed to free device array");
    }

    cuerr = cudaMalloc((void**)&d_y, n * sizeof(float));
    if (cuerr != cudaSuccess) {
        fprintf(stderr, "Failed to free device array");
    }

    cudaMemset(d_x, 1.0f, n);
    cudaMemset(d_y, 2.0f, n);
    is_alloc = 1;
}
#endif
/* vi:set sw=4 sts=4 tw=80: */
