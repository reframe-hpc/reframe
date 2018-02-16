program mpihel
implicit none

include 'mpif.h'

    integer :: rank, size, ierr
    integer :: mpiversion, mpisubversion
    integer :: resultlen = -1
    character(len=MPI_MAX_LIBRARY_VERSION_STRING) :: mpilibversion

    call MPI_INIT(ierr)
    call MPI_COMM_RANK(MPI_COMM_WORLD, rank, ierr)
    call MPI_COMM_SIZE(MPI_COMM_WORLD, size, ierr)
    write (*,'(A23,1X,I3,1X,A6,1X,I3,1X,A12,1X,I3,1X,A6,I3)') 'Hello World from thread', 0, &
     'out of', 1, 'from process', rank, 'out of', size

    call MPI_Get_version(mpiversion, mpisubversion, iErr)
    call MPI_Get_library_version(mpilibversion, resultlen, ierr)
    print *, '# MPI-',mpiversion,'.',mpisubversion,' = ',trim(mpilibversion)
    !flush(6)

    call MPI_FINALIZE(ierr)

end program mpihel
