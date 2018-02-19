#include <stdio.h>   
#include <omp.h>
 
int main(int argc, char *argv[])
{
  int tid, nthreads;
  #pragma omp parallel private(tid, nthreads)
  {
    tid = omp_get_thread_num();
    nthreads = omp_get_num_threads();
    #pragma omp critical
    {
      printf("Hello World from thread %d out of %d from process %d out of %d\n",
       tid, nthreads, 0, 1);
    }
  }
 
  return 0;
}
