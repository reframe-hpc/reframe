program openacc_cuda_mpi_cppstd
  ! This code tests MPI communication on GPU with OpenACC using the
  ! host_data directive + CUDA call from Fortran as well as C++ function 
  ! using std library call
  implicit none


  include 'mpif.h'
  integer :: status(MPI_STATUS_SIZE)

  integer :: ierr, i
  integer :: cpp_std_sum ! Sum done with C++ call to STD lib
  integer :: mpi_size, mpi_rank
  integer(8) :: mydata(1), data_sum(1), ref_val
  real, allocatable :: f1(:), f2(:), f3(:) 

  ! Test parameter
  integer, parameter :: NSIZE = 10
  real, parameter :: EXPECTED_CUDA_SUM = 110.0
  real, parameter :: EXPECTED_CPP_STD_SUM = 55.0

  call MPI_Init(ierr)
  call MPI_Comm_size(MPI_COMM_WORLD, mpi_size, ierr)
  call MPI_Comm_rank(MPI_COMM_WORLD, mpi_rank, ierr)
  mydata(1) = mpi_rank
#ifdef _OPENACC
  if (mpi_rank == 0) write(*,*) "MPI test on GPU with OpenACC using ",mpi_size,"tasks"
#else
  if (mpi_rank == 0) write(*,*) "MPI test on CPU using ",mpi_size,"tasks"
#endif

  !$acc data copy(mydata,data_sum)
  !$acc host_data use_device(mydata,data_sum)
  call MPI_Reduce(mydata, data_sum, 1, MPI_INTEGER8, MPI_SUM, 0, MPI_COMM_WORLD, ierr)
  !$acc end host_data
  !$acc end data


  if(mpi_rank == 0) then

    ! Allocate and initialize arrays on the GPU
    allocate(f1(NSIZE))
    allocate(f2(NSIZE))
    allocate(f3(NSIZE))

    !$acc data pcreate(f1,f2,f3)
    !$acc parallel loop  
    do i = 1, NSIZE
      f1(i) = i
      f2(i) = i
      f3(i) = i
    end do 
    !$acc update host(f1,f2,f3)
  
    ! Call a CUDA kernel with host arrays 
    call call_cuda_kernel_with_copy(f1, f2, NSIZE)

#ifdef _OPENACC
    ! Call a CUDA kernel without data copy, use device ptr
    call call_cuda_kernel_no_copy(f3, f2, NSIZE)
    !$acc update host(f3)
#endif

    ! Call a C++ function using STD lib
    call call_cpp_std(f2, NSIZE, cpp_std_sum)
    !$acc end data 
  end if

  !Check results
  if (mpi_rank == 0) then
    ref_val = 0
    do i = 0, mpi_size - 1
      ref_val = ref_val + i
    end do
    if (sum(f1) /= EXPECTED_CUDA_SUM) then
      write (*,*) "Result : FAIL"
      write (*,*) "Expected value sum(f1): ", EXPECTED_CUDA_SUM, "actual value:", sum(f1)
    else if (sum(f3) /= EXPECTED_CUDA_SUM) then
      write (*,*) "Result : FAIL"
      write (*,*) "Expected value sum(f3): ", EXPECTED_CUDA_SUM, "actual value:", sum(f3)
    else if (data_sum(1) /= ref_val) then
      write (*,*) "Result : FAIL"
      write (*,*) "Expected value data_sum: ", ref_val, "actual value:", data_sum(1)
    else if (cpp_std_sum /= EXPECTED_CPP_STD_SUM) then
      write (*,*) "Result : FAIL"
      write (*,*) "Expected value stdres: ", EXPECTED_CPP_STD_SUM, "actual value:", cpp_std_sum
    else
      write (*,*) "Result : OK"
    end if
  end if

  if(mpi_rank == 0) then
    deallocate(f1)
    deallocate(f2)
    deallocate(f3)
    write (*,*) "Result: OK"
  end if

  call MPI_Finalize(ierr);

contains
  subroutine call_cuda_kernel_with_copy(f1,f2,n)
    use, intrinsic :: iso_c_binding
    implicit none
    real, intent(inout), target :: f1(:)
    real, intent(in), target :: f2(:)
    integer, intent(in) :: n
  
    interface
      subroutine cuda_kernel_with_copy(f1,f2,n) bind(c,name='cuda_kernel_with_copy')
        use, intrinsic :: iso_c_binding
        type(c_ptr), intent(in), value :: f1, f2
        integer, intent(in), value :: n
      end subroutine cuda_kernel_with_copy
    end interface

    call cuda_kernel_with_copy(c_loc(f1(1)), c_loc(f2(1)), n)
  end subroutine call_cuda_kernel_with_copy

  subroutine call_cuda_kernel_no_copy(f1,f2,n)
    use, intrinsic :: iso_c_binding
    implicit none
    real, intent(inout), target :: f1(:)
    real, intent(in), target :: f2(:)
    integer, intent(in) :: n
  
    interface
      subroutine cuda_kernel_no_copy(f1,f2,n) bind(c,name='cuda_kernel_no_copy')
        use, intrinsic :: iso_c_binding
        type(c_ptr), intent(in), value :: f1, f2
        integer, intent(in), value :: n
      end subroutine cuda_kernel_no_copy
    end interface

    !$acc data present(f1, f2)
    !$acc host_data use_device(f1, f2)
    call cuda_kernel_no_copy(c_loc(f1(1)), c_loc(f2(1)), n)
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
end program openacc_cuda_mpi_cppstd
