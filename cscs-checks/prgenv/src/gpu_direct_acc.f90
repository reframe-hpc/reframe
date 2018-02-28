program GpuDirectAcc
    implicit none

    include 'mpif.h'

    integer :: ierr, status
    integer :: mpi_size, mpi_rank
    integer(8) :: mydata(1)

    call MPI_Init(ierr)

    call MPI_Comm_size(MPI_COMM_WORLD, mpi_size, ierr)
    call MPI_Comm_rank(MPI_COMM_WORLD, mpi_rank, ierr)

    if (mpi_size.ne.2) then
        if (mpi_rank.eq.0) write (*,*) "2 MPI ranks required"
        call MPI_Finalize(ierr);
        stop
    end if

    mydata(1) = mpi_rank

!$acc data copy(mydata)
    if (mpi_rank.eq.0) then
!$acc host_data use_device(mydata)
        call MPI_Recv(mydata, 1, MPI_INTEGER8, 1, 0, MPI_COMM_WORLD, status, ierr)
!$acc end host_data
    else
!$acc host_data use_device(mydata)
        call MPI_Send(mydata, 1, MPI_INTEGER8, 0, 0, MPI_COMM_WORLD, ierr)
!$acc end host_data
    end if
!$acc end data

    if (mpi_rank.eq.0) then
        write (*,*) "Result : ", mydata
    end if

    call MPI_Finalize(ierr);

end program GpuDirectAcc
