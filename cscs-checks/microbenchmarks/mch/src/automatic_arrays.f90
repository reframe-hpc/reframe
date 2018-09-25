! This code tests if the compilers implement a data pool for automatic arrays
module setup
  implicit none

  integer, parameter :: nvec = 20000
  integer, parameter :: niter = 10

  real*8, parameter :: max_rel_diff = 0.15

  public :: nvec, niter, max_rel_diff

end module

module workarrays
  implicit none
  real*8, allocatable :: a1(:), a2(:), a3(:), a4(:), a5(:), a6(:), a7(:),      &
                         a8(:), a9(:)
  real*8, allocatable :: aa1(:,:), aa2(:,:), aa3(:,:)
  real*8, allocatable :: zparam(:)

  public :: a1, a2, a3, a4, a5, a6, a7, a8, a9
  public :: aa1, aa2, aa3
  public :: zparam

  public :: allocate_arrays, deallocate_arrays

  contains

  subroutine allocate_arrays(nvec, lacc)
    integer, intent(in) :: nvec
    logical, intent(in) :: lacc

    allocate(a1(nvec), a2(nvec), a3(nvec), a4(nvec), a5(nvec), a6(nvec),        &
             a7(nvec), a8(nvec), a9(nvec))
    a1 = 0D0
    a2 = 0D0
    a3 = 0D0
    a4 = 0D0
    a5 = 0D0
    a6 = 0D0
    a7 = 0D0
    a8 = 0D0
    a9 = 0D0

    allocate(aa1(nvec,5), aa2(nvec,5), aa3(nvec,5))
    aa1 = 0D0
    aa2 = 0D0
    aa3 = 0D0

    allocate(zparam(8))
    zparam = 0D0

    !$acc enter data copyin( a1, a2, a3, a4, a5, a6, a7, a8, a9,               &
    !$acc                    aa1, aa2, aa3, zparam ) if( lacc )

  end subroutine allocate_arrays

  subroutine deallocate_arrays(lacc)
    logical, intent(in) :: lacc

    !$acc exit data delete( a1, a2, a3, a4, a5, a6, a7, a8, a9,                &
    !$acc                    aa1, aa2, aa3, zparam ) if( lacc )

    deallocate(a1, a2, a3, a4, a5, a6, a7, a8, a9)

    deallocate(aa1, aa2, aa3)

    deallocate(zparam)

  end subroutine deallocate_arrays

end module workarrays

module computation
  implicit none

  PUBLIC :: cpu_workarrays, cpu_automatic, gpu_workarrays, gpu_automatic
  contains

  subroutine cpu_workarrays(nvec,a,b)
    USE workarrays
    integer, intent(in)   :: nvec
    real*8, intent(inout) :: a(nvec)
    real*8, intent(in)    :: b(nvec)

    integer :: i, k, iparam, il

    do iparam=1,8
      zparam(iparam) = 0.1D0*iparam
    end do

    do i=1,nvec
      a1(i) = 0.1D0*(1.0D0+1.9D0/i)
      a2(i) = 0.2D0*(1.1D0+1.8D0/i)
      a3(i) = 0.3D0*(1.2D0+1.7D0/i)
      a4(i) = 0.4D0*(1.3D0+1.6D0/i)
      a5(i) = 0.5D0*(1.4D0+1.5D0/i)
      a6(i) = 0.6D0*(1.5D0+1.4D0/i)
      a7(i) = 0.7D0*(1.6D0+1.3D0/i)
      a8(i) = 0.8D0*(1.7D0+1.2D0/i)
      a9(i) = 0.9D0*(1.8D0+1.1D0/i)

      aa1(i,1) = 0.5D0*(1.0D0+(1.0D0-0.1D0)/i)
      aa1(i,2) = 0.5D0*(1.0D0+(1.0D0-0.2D0)/i)
      aa1(i,3) = 0.5D0*(1.0D0+(1.0D0-0.3D0)/i)
      aa1(i,4) = 0.5D0*(1.0D0+(1.0D0-0.4D0)/i)
      aa1(i,5) = 0.5D0*(1.0D0+(1.0D0-0.5D0)/i)

      aa2(i,1) = 0.7D0*(1.0D0+(1.0D0+0.1D0)/i)
      aa2(i,2) = 0.7D0*(1.0D0+(1.0D0+0.2D0)/i)
      aa2(i,3) = 0.7D0*(1.0D0+(1.0D0+0.3D0)/i)
      aa2(i,4) = 0.7D0*(1.0D0+(1.0D0+0.4D0)/i)
      aa2(i,5) = 0.7D0*(1.0D0+(1.0D0+0.5D0)/i)

      aa3(i,1) = 0.9D0*(1.0D0+(1.0D0-0.9D0)/i)
      aa3(i,2) = 0.9D0*(1.0D0+(1.0D0-0.8D0)/i)
      aa3(i,3) = 0.9D0*(1.0D0+(1.0D0-0.7D0)/i)
      aa3(i,4) = 0.9D0*(1.0D0+(1.0D0-0.6D0)/i)
      aa3(i,5) = 0.9D0*(1.0D0+(1.0D0-0.5D0)/i)
    end do

    do i=1,nvec
      do iparam=1,8 ! to make more operations
        a(i) = zparam(iparam)*(1.0D0+cos(a(i))) + b(i)*(1.0D0 + sin(1.0D0      &
               +a1(i)+a2(i)+a3(i)+a4(i)+a5(i)+a6(i)+a7(i)+a8(i)+a9(i)          &
               +aa1(i,1)+aa1(i,2)+aa1(i,3)+aa1(i,4)+aa1(i,5)                   &
               +aa2(i,1)+aa2(i,2)+aa2(i,3)+aa2(i,4)+aa2(i,5)                   &
               +aa3(i,1)+aa3(i,2)+aa3(i,3)+aa3(i,4)+aa3(i,5)))
      end do
    end do !i

  end subroutine cpu_workarrays

  subroutine cpu_automatic(nvec,a,b)
    integer, intent(in)   :: nvec
    real*8, intent(inout) :: a(nvec)
    real*8, intent(in)    :: b(nvec)

    integer :: i, k, iparam, il
    real*8 :: a1(nvec), a2(nvec), a3(nvec), a4(nvec), a5(nvec), a6(nvec),      &
              a7(nvec), a8(nvec), a9(nvec)
    real*8 :: aa1(nvec,5), aa2(nvec,5), aa3(nvec,5)
    real*8 :: zparam(8)

    do iparam=1,8
      zparam(iparam) = 0.1D0*iparam
    end do

    do i=1,nvec
      a1(i) = 0.1D0*(1.0D0+1.9D0/i)
      a2(i) = 0.2D0*(1.1D0+1.8D0/i)
      a3(i) = 0.3D0*(1.2D0+1.7D0/i)
      a4(i) = 0.4D0*(1.3D0+1.6D0/i)
      a5(i) = 0.5D0*(1.4D0+1.5D0/i)
      a6(i) = 0.6D0*(1.5D0+1.4D0/i)
      a7(i) = 0.7D0*(1.6D0+1.3D0/i)
      a8(i) = 0.8D0*(1.7D0+1.2D0/i)
      a9(i) = 0.9D0*(1.8D0+1.1D0/i)

      aa1(i,1) = 0.5D0*(1.0D0+(1.0D0-0.1D0)/i)
      aa1(i,2) = 0.5D0*(1.0D0+(1.0D0-0.2D0)/i)
      aa1(i,3) = 0.5D0*(1.0D0+(1.0D0-0.3D0)/i)
      aa1(i,4) = 0.5D0*(1.0D0+(1.0D0-0.4D0)/i)
      aa1(i,5) = 0.5D0*(1.0D0+(1.0D0-0.5D0)/i)

      aa2(i,1) = 0.7D0*(1.0D0+(1.0D0+0.1D0)/i)
      aa2(i,2) = 0.7D0*(1.0D0+(1.0D0+0.2D0)/i)
      aa2(i,3) = 0.7D0*(1.0D0+(1.0D0+0.3D0)/i)
      aa2(i,4) = 0.7D0*(1.0D0+(1.0D0+0.4D0)/i)
      aa2(i,5) = 0.7D0*(1.0D0+(1.0D0+0.5D0)/i)

      aa3(i,1) = 0.9D0*(1.0D0+(1.0D0-0.9D0)/i)
      aa3(i,2) = 0.9D0*(1.0D0+(1.0D0-0.8D0)/i)
      aa3(i,3) = 0.9D0*(1.0D0+(1.0D0-0.7D0)/i)
      aa3(i,4) = 0.9D0*(1.0D0+(1.0D0-0.6D0)/i)
      aa3(i,5) = 0.9D0*(1.0D0+(1.0D0-0.5D0)/i)
    end do

    do i=1,nvec
      do iparam=1,8 ! to make more operations
        a(i) = zparam(iparam)*(1.0D0+cos(a(i))) + b(i)*(1.0D0 + sin(1.0D0      &
               +a1(i)+a2(i)+a3(i)+a4(i)+a5(i)+a6(i)+a7(i)+a8(i)+a9(i)          &
               +aa1(i,1)+aa1(i,2)+aa1(i,3)+aa1(i,4)+aa1(i,5)                   &
               +aa2(i,1)+aa2(i,2)+aa2(i,3)+aa2(i,4)+aa2(i,5)                   &
               +aa3(i,1)+aa3(i,2)+aa3(i,3)+aa3(i,4)+aa3(i,5)))
      end do
    end do

  end subroutine cpu_automatic

  subroutine gpu_workarrays(nvec,a,b)
    USE workarrays
    integer, intent(in)   :: nvec
    real*8, intent(inout) :: a(nvec)
    real*8, intent(in)    :: b(nvec)

    integer :: i, k, iparam, il

    !$acc data present( a, b,                                                  &
    !$acc               a1, a2, a3, a4, a5, a6, a7, a8, a9,                    &
    !$acc               aa1, aa2, aa3,                                         &
    !$acc               zparam )

    !$acc parallel
    !$acc loop gang vector
    do iparam=1,8
      zparam(iparam) = 0.1D0*iparam
    end do
    !$acc end parallel

    !$acc parallel
    !$acc loop gang vector
    do i=1,nvec
      a1(i) = 0.1D0*(1.0D0+1.9D0/i)
      a2(i) = 0.2D0*(1.1D0+1.8D0/i)
      a3(i) = 0.3D0*(1.2D0+1.7D0/i)
      a4(i) = 0.4D0*(1.3D0+1.6D0/i)
      a5(i) = 0.5D0*(1.4D0+1.5D0/i)
      a6(i) = 0.6D0*(1.5D0+1.4D0/i)
      a7(i) = 0.7D0*(1.6D0+1.3D0/i)
      a8(i) = 0.8D0*(1.7D0+1.2D0/i)
      a9(i) = 0.9D0*(1.8D0+1.1D0/i)

      aa1(i,1) = 0.5D0*(1.0D0+(1.0D0-0.1D0)/i)
      aa1(i,2) = 0.5D0*(1.0D0+(1.0D0-0.2D0)/i)
      aa1(i,3) = 0.5D0*(1.0D0+(1.0D0-0.3D0)/i)
      aa1(i,4) = 0.5D0*(1.0D0+(1.0D0-0.4D0)/i)
      aa1(i,5) = 0.5D0*(1.0D0+(1.0D0-0.5D0)/i)

      aa2(i,1) = 0.7D0*(1.0D0+(1.0D0+0.1D0)/i)
      aa2(i,2) = 0.7D0*(1.0D0+(1.0D0+0.2D0)/i)
      aa2(i,3) = 0.7D0*(1.0D0+(1.0D0+0.3D0)/i)
      aa2(i,4) = 0.7D0*(1.0D0+(1.0D0+0.4D0)/i)
      aa2(i,5) = 0.7D0*(1.0D0+(1.0D0+0.5D0)/i)

      aa3(i,1) = 0.9D0*(1.0D0+(1.0D0-0.9D0)/i)
      aa3(i,2) = 0.9D0*(1.0D0+(1.0D0-0.8D0)/i)
      aa3(i,3) = 0.9D0*(1.0D0+(1.0D0-0.7D0)/i)
      aa3(i,4) = 0.9D0*(1.0D0+(1.0D0-0.6D0)/i)
      aa3(i,5) = 0.9D0*(1.0D0+(1.0D0-0.5D0)/i)
    end do
    !$acc end parallel

    !$acc parallel
    !$acc loop gang vector
    do i=1,nvec
      !$acc loop seq
      do iparam=1,8 ! to make more operations
        a(i) = zparam(iparam)*(1.0D0+cos(a(i))) + b(i)*(1.0D0 + sin(1.0D0      &
               +a1(i)+a2(i)+a3(i)+a4(i)+a5(i)+a6(i)+a7(i)+a8(i)+a9(i)          &
               +aa1(i,1)+aa1(i,2)+aa1(i,3)+aa1(i,4)+aa1(i,5)                   &
               +aa2(i,1)+aa2(i,2)+aa2(i,3)+aa2(i,4)+aa2(i,5)                   &
               +aa3(i,1)+aa3(i,2)+aa3(i,3)+aa3(i,4)+aa3(i,5)))
      end do
    end do
    !$acc end parallel

    !$acc end data

  end subroutine gpu_workarrays

  subroutine gpu_automatic(nvec,a,b)
    integer, intent(in)   :: nvec
    real*8, intent(inout) :: a(nvec)
    real*8, intent(in)    :: b(nvec)

    integer :: i, k, iparam, il
    real*8 :: a1(nvec), a2(nvec), a3(nvec), a4(nvec), a5(nvec), a6(nvec),      &
              a7(nvec), a8(nvec), a9(nvec)
    real*8 :: aa1(nvec,5), aa2(nvec,5), aa3(nvec,5)
    real*8 :: zparam(8)

    !$acc data present( a, b )                                                 &
    !$acc      create( a1, a2, a3, a4, a5, a6, a7, a8, a9,                     &
    !$acc              aa1, aa2, aa3,                                          &
    !$acc              zparam )

    !$acc parallel
    !$acc loop gang vector
    do iparam=1,8
      zparam(iparam) = 0.1D0*iparam
    end do
    !$acc end parallel

    !$acc parallel
    !$acc loop gang vector
    do i=1,nvec
      a1(i) = 0.1D0*(1.0D0+1.9D0/i)
      a2(i) = 0.2D0*(1.1D0+1.8D0/i)
      a3(i) = 0.3D0*(1.2D0+1.7D0/i)
      a4(i) = 0.4D0*(1.3D0+1.6D0/i)
      a5(i) = 0.5D0*(1.4D0+1.5D0/i)
      a6(i) = 0.6D0*(1.5D0+1.4D0/i)
      a7(i) = 0.7D0*(1.6D0+1.3D0/i)
      a8(i) = 0.8D0*(1.7D0+1.2D0/i)
      a9(i) = 0.9D0*(1.8D0+1.1D0/i)

      aa1(i,1) = 0.5D0*(1.0D0+(1.0D0-0.1D0)/i)
      aa1(i,2) = 0.5D0*(1.0D0+(1.0D0-0.2D0)/i)
      aa1(i,3) = 0.5D0*(1.0D0+(1.0D0-0.3D0)/i)
      aa1(i,4) = 0.5D0*(1.0D0+(1.0D0-0.4D0)/i)
      aa1(i,5) = 0.5D0*(1.0D0+(1.0D0-0.5D0)/i)

      aa2(i,1) = 0.7D0*(1.0D0+(1.0D0+0.1D0)/i)
      aa2(i,2) = 0.7D0*(1.0D0+(1.0D0+0.2D0)/i)
      aa2(i,3) = 0.7D0*(1.0D0+(1.0D0+0.3D0)/i)
      aa2(i,4) = 0.7D0*(1.0D0+(1.0D0+0.4D0)/i)
      aa2(i,5) = 0.7D0*(1.0D0+(1.0D0+0.5D0)/i)

      aa3(i,1) = 0.9D0*(1.0D0+(1.0D0-0.9D0)/i)
      aa3(i,2) = 0.9D0*(1.0D0+(1.0D0-0.8D0)/i)
      aa3(i,3) = 0.9D0*(1.0D0+(1.0D0-0.7D0)/i)
      aa3(i,4) = 0.9D0*(1.0D0+(1.0D0-0.6D0)/i)
      aa3(i,5) = 0.9D0*(1.0D0+(1.0D0-0.5D0)/i)
    end do
    !$acc end parallel

    !$acc parallel
    !$acc loop gang vector
    do i=1,nvec
      !$acc loop seq
      do iparam=1,8 ! to make more operations
        a(i) = zparam(iparam)*(1.0D0+cos(a(i))) + b(i)*(1.0D0 + sin(1.0D0      &
               +a1(i)+a2(i)+a3(i)+a4(i)+a5(i)+a6(i)+a7(i)+a8(i)+a9(i)          &
               +aa1(i,1)+aa1(i,2)+aa1(i,3)+aa1(i,4)+aa1(i,5)                   &
               +aa2(i,1)+aa2(i,2)+aa2(i,3)+aa2(i,4)+aa2(i,5)                   &
               +aa3(i,1)+aa3(i,2)+aa3(i,3)+aa3(i,4)+aa3(i,5)))
      end do
    end do
    !$acc end parallel

    !$acc end data

  end subroutine gpu_automatic

end module computation

program AutomaticArrays
  use setup, only: nvec, niter, max_rel_diff
  use computation
  use workarrays, only: allocate_arrays, deallocate_arrays
  implicit none
  include 'mpif.h'

  real*8, allocatable :: a(:), b(:), ref_a(:)

  integer :: ierr
  integer :: mpi_size, mpi_rank
  integer :: nt, i
  real*8 :: walltime(4,niter)
  real*8 :: error(4)

  logical :: validated, success_cpu, success_gpu
  real*8 :: mean_time(4), benchmark_cpu, benchmark_gpu

  call MPI_Init(ierr)
  call MPI_Comm_size(MPI_COMM_WORLD, mpi_size, ierr)
  call MPI_Comm_rank(MPI_COMM_WORLD, mpi_rank, ierr)

  !----------------------------------------------------------------------------!
  ! CPU reference with work arrays

  allocate(a(nvec), b(nvec), ref_a(nvec))
  call allocate_arrays(nvec, .false.)

  do nt=1,niter
    do i=1,nvec
      a(i) = 0.0D0
      b(i) = 0.1D0
    end do

    walltime(1,nt) = MPI_WTIME()
    call cpu_workarrays(nvec,a,b)
    walltime(1,nt) = MPI_WTIME() - walltime(1,nt)
  end do

  ref_a = a
  error(1) = sum(ref_a - a)

  call deallocate_arrays(.false.)
  deallocate(a, b)

  !----------------------------------------------------------------------------!
  ! CPU reference with automatic arrays

  allocate(a(nvec), b(nvec))

  do nt=1,niter
    do i=1,nvec
      a(i) = 0.0D0
      b(i) = 0.1D0
    end do

    walltime(2,nt) = MPI_WTIME()
    call cpu_automatic(nvec,a,b)
    walltime(2,nt) = MPI_WTIME() - walltime(2,nt)
  end do

  error(2) = sum(ref_a - a)

  deallocate(a, b)

  !----------------------------------------------------------------------------!
  ! GPU dummy calculcations
  allocate(a(nvec), b(nvec))
  !$acc data create(a, b)
  !$acc parallel
  !$acc loop gang vector
  do i=1,nvec
    a(i) = 1.3*i**0.5
    b(i) = 0.3*i**0.75
  end do
  !$acc end parallel

  !$acc parallel
  !$acc loop seq
  do nt=1, 10
    !$acc loop gang vector
    do i=1,nvec
      b(i) = a(i)*a(i)
    end do
  end do
  !$acc end parallel

  !$acc end data
  deallocate(a, b)

  !----------------------------------------------------------------------------!
  ! GPU OpenACC with work arrays

  allocate(a(nvec), b(nvec))
  call allocate_arrays(nvec, .true.)

  !$acc data create( a, b )

  do nt=1,niter
    !$acc parallel
    !$acc loop gang vector
    do i=1,nvec
      a(i) = 0.0D0
      b(i) = 0.1D0
    end do
    !$acc end parallel

    !$acc wait
    walltime(3,nt) = MPI_WTIME()
    call gpu_workarrays(nvec,a,b)
    !$acc wait
    walltime(3,nt) = MPI_WTIME() - walltime(3,nt)
  end do

  !$acc update host( a )

  !$acc end data

  error(3) = sum(ref_a - a)

  call deallocate_arrays(.true.)
  deallocate(a, b)

  !----------------------------------------------------------------------------!
  ! GPU OpenACC with automatic arrays

  allocate(a(nvec), b(nvec))

  !$acc data create( a, b )

  do nt=1,niter
    !$acc parallel
    !$acc loop gang vector
    do i=1,nvec
      a(i) = 0.0D0
      b(i) = 0.1D0
    end do
    !$acc end parallel

    !$acc wait
    walltime(4,nt) = MPI_WTIME()
    call gpu_automatic(nvec,a,b)
    !$acc wait
    walltime(4,nt) = MPI_WTIME() - walltime(4,nt)
  end do

  !$acc update host( a )

  !$acc end data

  error(4) = sum(ref_a - a)

  deallocate(a, b)

  !----------------------------------------------------------------------------!
  ! Check results

  ! Check if calculations are correct
  if (any(error > 1e-10)) then
    validated = .false.
    write (*,*) "Calculations did not validate and timings are meaningless."
    write (*,*) "Absolute errors: ", error
  else
    validated = .true.
  end if

  ! Check timings
  if (validated) then
    ! Check CPU timing
    mean_time(1) = sum(walltime(1, 2:))/(niter - 1)
    mean_time(2) = sum(walltime(2, 2:))/(niter - 1)
    benchmark_cpu = (mean_time(2) - mean_time(1))/mean_time(2)
    if (benchmark_cpu > max_rel_diff) then
      success_cpu = .false.
      write (*,'(A)') "Compiler doesn't implement data pool for CPU!"
      write (*,'(A,F5.1,A,F5.1,A)') "Relative difference is too large: ",      &
                                     benchmark_cpu*100, "% > ", max_rel_diff*100, "%"
    else
      success_cpu = .true.
    end if
    if (.not. success_cpu) then
      write (*,'(A,ES9.3,A)') "CPU work arrays timing: ", mean_time(1), " s"
      write (*,'(A,ES9.3,A)') "CPU automatic arrays timing: ", mean_time(2), " s"
      write (*,'(A,F5.1,A)') "CPU automatic arrays relative timing: ", benchmark_cpu*100, "%"
      write (*,*) ""
    end if

    ! Check GPU timing
    mean_time(3) = sum(walltime(3, 2:))/(niter - 1)
    mean_time(4) = sum(walltime(4, 2:))/(niter - 1)
    benchmark_gpu = (mean_time(4) - mean_time(3))/mean_time(4)
    if (benchmark_gpu > max_rel_diff) then
      success_gpu = .false.
      write (*,'(A)') "Compiler doesn't implement data pool for GPU!"
      write (*,'(A,F5.1,A,F5.1,A)') "Relative difference is too large: ",      &
                                     benchmark_gpu*100, "% > ", max_rel_diff*100, "%"
    else
      success_gpu = .true.
    end if
    if (.not. success_gpu) then
      write (*,'(A,ES9.3,A)') "GPU work arrays timing: ", mean_time(3), " s"
      write (*,'(A,ES9.3,A)') "GPU automatic arrays timing: ", mean_time(4), " s"
      write (*,'(A,F5.1,A)') "GPU automatic arrays relative timing: ", benchmark_gpu*100, "%"
      write (*,*) ""
    end if
  end if

  write (*,'(A,ES9.3,A)') "Timing: ", mean_time(4), " s"
  if (success_gpu) then
    write (*,'(A)') "Result: OK"
  else
    write (*,'(A)') "Result: FAIL"
  end if

  call MPI_Finalize(ierr)

end program AutomaticArrays
