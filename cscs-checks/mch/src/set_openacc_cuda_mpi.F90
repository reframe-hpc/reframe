! This code tests MPI tasks communication with GPU devices 
! using OpenACC directives and setting one device per task
program set_openacc_cuda_mpi
  use openacc
  implicit none

  include 'mpif.h'
#ifdef CRAY
  integer, parameter :: ACC_DEVICE_TYPE = 8
#else
  integer, parameter :: ACC_DEVICE_TYPE = 4
#endif
  integer, parameter :: ARRAYSIZE = 10
  integer(kind=ACC_DEVICE_TYPE) :: devicetype
  integer :: status(MPI_STATUS_SIZE), mpi_size, mpi_rank
  integer :: ierr, i, gpuid, ngpus, localsum(2), globalsum(2)
  real, allocatable :: array1(:), array2(:) 
  
  call MPI_Init(ierr)
  call MPI_Comm_size(MPI_COMM_WORLD, mpi_size, ierr)
  call MPI_Comm_rank(MPI_COMM_WORLD, mpi_rank, ierr)

! each task creates two different arrays: the sum of their elements will be 10*mpi_rank
  allocate(array1(ARRAYSIZE))
  allocate(array2(ARRAYSIZE))

#ifdef _OPENACC
  if (mpi_rank == 0) then 
    devicetype = acc_get_device_type()
    ngpus = acc_get_num_devices(devicetype)
    write(*,*) "MPI test with OpenACC using", mpi_size, "tasks and ", ngpus-1, "GPU devices"
    do i = 1, ARRAYSIZE
     array1(i) = .0
     array2(i) = .0
    end do
  else
   ! each task different from 0 addresses a different GPU device
   call acc_set_device_num(mpi_rank, acc_device_nvidia)
   call acc_init(acc_device_nvidia)
   gpuid = acc_get_device_num(devicetype)
   write(*,*) "MPI task ", mpi_rank, "is using GPU id ", gpuid

   !$acc data pcreate(array1,array2)
   !$acc parallel loop  
   do i = 1, ARRAYSIZE
     array1(i) = mpi_rank*0.25
     array2(i) = mpi_rank*0.75
   end do
   !$acc update host(array1,array2)

! the current mpi_rank computes localsum(1)
!   localsum(1) = sum(array1)+sum(array2)
   call call_cpp_std(array1, ARRAYSIZE, localsum(1))
   call call_cpp_std(array2, ARRAYSIZE, localsum(1))

   ! compute the sum of the arrays on the GPU calling a CUDA kernel using device ptr 
   call call_cuda_kernel_no_copy(array1, array2, ARRAYSIZE)
   !$acc update host(array1)
   !$acc end data

! array1 is now equal to sum(array1)+sum(array2): compute localsum(2)
   localsum(2) = sum(array1)
  end if
#endif

! the current mpi_rank sends localsum to compute globalsum over all mpi tasks
  call MPI_Reduce(localsum, globalsum, 2, MPI_INTEGER, MPI_SUM, 0, MPI_COMM_WORLD, ierr)

 if(mpi_rank == 0) then
    if (globalsum(1) == globalsum(2)) then
      write (*,*) "CPU sum : ", globalsum(1), " GPU sum : ", globalsum(2)
      write (*,*) "Test Result : OK"
    else
      write (*,*) "CPU sum : ", globalsum(1), " GPU sum : ", globalsum(2)
      write (*,*) "Test Result : FAIL"
    end if
  end if

  deallocate(array1)
  deallocate(array2)
  call MPI_Finalize(ierr);


contains
  subroutine call_cuda_kernel_with_copy(a,b,n)
    use, intrinsic :: iso_c_binding
    implicit none
    real, intent(inout), target :: a(:)
    real, intent(in), target :: b(:)
    integer, intent(in) :: n
  
    interface
      subroutine cuda_kernel_with_copy(a,b,n) bind(c,name='cuda_kernel_with_copy')
        use, intrinsic :: iso_c_binding
        type(c_ptr), intent(in), value :: a, b
        integer, intent(in), value :: n
      end subroutine cuda_kernel_with_copy
    end interface

    call cuda_kernel_with_copy(c_loc(a(1)), c_loc(b(1)), n)
  end subroutine call_cuda_kernel_with_copy

  subroutine call_cuda_kernel_no_copy(a,b,n)
    use, intrinsic :: iso_c_binding
    implicit none
    real, intent(inout), target :: a(:)
    real, intent(in), target :: b(:)
    integer, intent(in) :: n
  
    interface
      subroutine cuda_kernel_no_copy(a,b,n) bind(c,name='cuda_kernel_no_copy')
        use, intrinsic :: iso_c_binding
        type(c_ptr), intent(in), value :: a, b
        integer, intent(in), value :: n
      end subroutine cuda_kernel_no_copy
    end interface

    !$acc data present(a, b)
    !$acc host_data use_device(a, b)
    call cuda_kernel_no_copy(c_loc(a(1)), c_loc(b(1)), n)
    !$acc end host_data
    !$acc end data
  end subroutine call_cuda_kernel_no_copy

  subroutine call_cpp_std(f,n,i)
    use, intrinsic :: iso_c_binding
    implicit none
    real(kind=c_float), intent(in), target :: f(:)
    real(kind=c_float), pointer :: fp(:)
    integer, intent(in) :: n
    integer(kind=c_int), intent(out) :: i

    interface
      subroutine cpp_call(f,n,i) bind(c,name='do_smth_with_std')
        use, intrinsic :: iso_c_binding
        type(c_ptr), intent(in), value :: f
        integer, intent(in), value :: n
        integer(kind=c_int), intent(out) :: i
      end subroutine cpp_call
    end interface

    fp => f

    call cpp_call(c_loc(fp(1)), n, i)
  end subroutine call_cpp_std

end program set_openacc_cuda_mpi
