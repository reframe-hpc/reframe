program hello90
use omp_lib
integer:: id, nthreads
  !$omp parallel private(id, nthreads)
  id = omp_get_thread_num()
  nthreads = omp_get_num_threads()
  write (*,'(A23,1X,I3,1X,A6,1X,I3,1X,A12,1X,I3,1X,A6,I3)') 'Hello World from thread', id, &
	'out of', nthreads, 'from process', 0, 'out of', 1

  !$omp end parallel
end program

