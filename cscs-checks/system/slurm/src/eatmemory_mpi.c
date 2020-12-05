// MPI version of eatmemory.c from Julio Viera
// 12/2020: add cscs_read_proc_meminfo from jg (cscs)
#include <ctype.h>
#include <mpi.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define PROC_FILE "/proc/meminfo"
#define MEMTOTAL 0
#define MEMFREE 1
#define MEMCACHED 2
#define SWAPTOTAL 3
#define SWAPFREE 4
#define SWAPCACHED 5
#define MEMAVAIL 6
#define MEMORY_PERCENTAGE

typedef struct {
  char *str;
  uint32_t val;
} meminfo_t;

int cscs_read_proc_meminfo(int);

#ifdef MEMORY_PERCENTAGE
size_t getTotalSystemMemory() {
  long pages = sysconf(_SC_PHYS_PAGES);
  long page_size = sysconf(_SC_PAGE_SIZE);
  return pages * page_size;
}

size_t getFreeSystemMemory() {
  long pages = sysconf(_SC_AVPHYS_PAGES);
  long page_size = sysconf(_SC_PAGE_SIZE);
  return pages * page_size;
}
#endif

bool eat(long total, int chunk) {
  long i;
  int rank, mpi_size;
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  for (i = 0; i < total; i += chunk) {
    if (rank == 0) {
      int mb_mpi = chunk / 1048576;
      printf("Eating %d MB/mpi *%dmpi = -%d MB ", mb_mpi, mpi_size,
             mb_mpi * mpi_size);
      cscs_read_proc_meminfo(i);
    }
    short *buffer = malloc(sizeof(char) * chunk);
    if (buffer == NULL) {
      return false;
    }
    memset(buffer, 0, chunk);
  }
  return true;
}

int cscs_read_proc_meminfo(int i) {
  FILE *fp;
  meminfo_t meminfo[] = {{"MemTotal:", 0},     {"MemFree:", 0},
                         {"Cached:", 0},       {"SwapCached:", 0},
                         {"SwapTotal:", 0},    {"SwapFree:", 0},
                         {"MemAvailable:", 0}, {NULL, 0}};
  fp = fopen(PROC_FILE, "r");
  if (!fp) {
    printf("Cannot read %s", PROC_FILE);
    return -1;
  }
  char buf[80];
  while (fgets(buf, sizeof(buf), fp)) {
    int i;
    for (i = 0; meminfo[i].str; i++) {
      size_t len = strlen(meminfo[i].str);
      if (!strncmp(buf, meminfo[i].str, len)) {
        char *ptr = buf + len + 1;
        while (isspace(*ptr))
          ptr++;
        sscanf(ptr, "%u kB", &meminfo[i].val);
      }
    }
  }
  fclose(fp);

  printf("memory from %s: total: %u GB, free: %u GB, avail: %u GB, using: %u GB\n",
         PROC_FILE,
         meminfo[MEMTOTAL].val / 1048576, meminfo[MEMFREE].val / 1048576,
         meminfo[MEMAVAIL].val / 1048576,
         (meminfo[MEMTOTAL].val - meminfo[MEMAVAIL].val) / 1048576);
  return 0;
}

int main(int argc, char *argv[]) {
  int rank, mpi_size;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
#ifdef MEMORY_PERCENTAGE
  if (rank == 0) {
    printf("memory from sysconf: total: %zd avail: %zd\n", \
        getTotalSystemMemory(), getFreeSystemMemory() );
  }
#endif
  int i;
  for (i = 0; i < argc; i++) {
    char *arg = argv[i];
    if (strcmp(arg, "-h") == 0 || strcmp(arg, "-?") == 0 || argc == 1) {
      printf("Usage: eatmemory <size>\n");
      printf("Size can be specified in megabytes or gigabytes in the following "
             "way:\n");
      printf("#          # Bytes      example: 1024\n");
      printf("#M         # Megabytes  example: 15M\n");
      printf("#G         # Gigabytes  example: 2G\n");
#ifdef MEMORY_PERCENTAGE
      printf("#%%         # Percent    example: 50%%\n");
#endif
      printf("\n");
    } else if (i > 0) {
      int len = strlen(arg);
      char unit = arg[len - 1];
      long size = -1;
      int chunk =  33554432; //  32M
      // int chunk =  67108864; //  64M
      // int chunk = 134217728; // 128M
      // int chunk = 268435456; // = 256M
      // int chunk=536870912; // = 512M
      // int chunk=1073741824; // = 1G
      if (!isdigit(unit)) {
        if (unit == 'M' || unit == 'G') {
          arg[len - 1] = 0;
          size = atol(arg) * (unit == 'M' ? 1024 * 1024 : 1024 * 1024 * 1024);
        }
#ifdef MEMORY_PERCENTAGE
        else if (unit == '%') {
          size = (atol(arg) * (long)getFreeSystemMemory()) / 100;
        }
#endif
        else {
          printf("Invalid size format\n");
          exit(0);
        }
      } else {
        size = atoi(arg);
      }

      if (rank == 0) {
        cscs_read_proc_meminfo(i);
        printf("Peak: %d mpi * %ld bytes = %ld Mbytes\n", mpi_size, size,
               mpi_size * size / 1000000);
        printf("Eating %ld bytes in chunks of %d...\n", size, chunk);
        printf("Eating %ld (1byte=8bits) Mbytes in chunks of %d Kbytes\n",
               (size / 1000000), (chunk / 1000));
      }
      if (eat(size, chunk)) {
        if (isatty(fileno(stdin))) {
          printf("Done, press any key to free the memory\n");
        } else {
          if (rank == 0)
            printf("rank %d Done, kill this process to free the memory\n",
                   rank);
          while (true) {
            sleep(1);
          }
        }
      } else {
        printf("ERROR: Could not allocate the memory");
      }
    }
  }

  MPI_Finalize();
  return 0;
}
