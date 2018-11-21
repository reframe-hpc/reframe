program GpuDirectAcc
    ! This code tests MPI communication on GPU with OpenACC using the
    ! host_data directive
    implicit none

    include 'mpif.h'

    integer :: ierr, status(MPI_STATUS_SIZE), i
    integer :: mpi_size, mpi_rank
    integer(8) :: mydata(1), data_sum(1), ref_val

    call MPI_Init(ierr)

    call MPI_Comm_size(MPI_COMM_WORLD, mpi_size, ierr)
    call MPI_Comm_rank(MPI_COMM_WORLD, mpi_rank, ierr)

    mydata(1) = mpi_rank
#ifdef _OPENACC
    if (mpi_rank==0) write(*,*) "MPI test on GPU with OpenACC using ",mpi_size,"tasks"
#else
    if (mpi_rank==0) write(*,*) "MPI test on CPU using ",mpi_size,"tasks"
#endif

    !$acc data copy(mydata,data_sum)
    !$acc host_data use_device(mydata,data_sum)
    call MPI_Reduce(mydata, data_sum, 1, MPI_INTEGER8, MPI_SUM, 0, MPI_COMM_WORLD, ierr)
    !$acc end host_data
    !$acc end data

    !Check results
    if (mpi_rank.eq.0) then
      ref_val=0
      do i=0,mpi_size-1
        ref_val=ref_val+i
      end do
      if (data_sum(1)/=ref_val) then
        write (*,*) "Result : FAIL"
        write (*,*) "Expected value : ", ref_val, "actual value:",data_sum(1)
      else
        write (*,*) "Result : OK"
      end if
    end if

    call MPI_Finalize(ierr);

end program GpuDirectAcc
