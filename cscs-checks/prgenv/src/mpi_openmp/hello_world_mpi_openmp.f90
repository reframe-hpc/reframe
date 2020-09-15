program hello_world_mpi_openmp
use omp_lib
implicit none

include 'mpif.h'

integer :: rank, size, ierr, i, j, k, id, nthreads
integer, dimension(MPI_STATUS_SIZE) :: status

integer, allocatable :: buffer(:), output(:)

call MPI_INIT(ierr)
call MPI_COMM_RANK(MPI_COMM_WORLD, rank, ierr)
call MPI_COMM_SIZE(MPI_COMM_WORLD, size, ierr)

!$OMP PARALLEL SHARED(nthreads)
    !omp critical
      nthreads = omp_get_num_threads()
    !omp end critical
!$OMP END PARALLEL

allocate(buffer(nthreads))
allocate(output(nthreads*size))

! Populating the buffer with -1, this will be used for checking
buffer = -1
output = -1

! Populating the buffer
!$OMP PARALLEL PRIVATE(id, nthreads) SHARED(BUFFER, OUTPUT)
      id = omp_get_thread_num()
      buffer(id+1) = id
!$OMP END PARALLEL

CALL MPI_Gather(buffer, nthreads, MPI_INTEGER, &
                output, nthreads, MPI_INTEGER, &
                0, &
                MPI_COMM_WORLD, ierr)

if (rank .eq. 0) then
    do i = 1, size
       do j = 1, nthreads
         k = (nthreads * (i - 1)) +j
         write (*,'(A23,1X,I3,1X,A6,1X,I3,1X,A12,1X,I3,1X,A6,I3)') 'Hello World from thread', output(k), &
                  'out of', nthreads, 'from process', i-1, 'out of', size
       enddo
    enddo
endif

deallocate(buffer)
deallocate(output)

call MPI_FINALIZE(ierr)

end program hello_world_mpi_openmp