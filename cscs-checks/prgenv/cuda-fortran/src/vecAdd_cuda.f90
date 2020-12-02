module math_kernels
contains
  attributes(global) subroutine vadd(a, b, c)
    implicit none
    real(8) :: a(:), b(:), c(:)
    integer :: i, n
    n = size(a)
    i = blockDim%x * (blockIdx%x - 1) + threadIdx%x
    if (i <= n) c(i) = a(i) + b(i)
  end subroutine vadd
end module math_kernels

program main
  use math_kernels
  use cudafor
  implicit none

  ! Size of vectors
  integer, parameter :: n = 100000

  ! Input vectors
  real(8),dimension(n) :: a
  real(8),dimension(n) :: b
  ! Output vector
  real(8),dimension(n) :: c
  ! Input vectors
  real(8),device,dimension(n) :: a_d
  real(8),device,dimension(n) :: b_d
  ! Output vector
  real(8),device,dimension(n) :: c_d
  type(dim3) :: grid, tBlock

  integer :: i
  real(8) :: vsum

  ! Initialize content of input vectors, vector a[i] = sin(i)^2 vector b[i] = cos(i)^2
  do i=1,n
     a(i) = sin(i*1D0)*sin(i*1D0)
     b(i) = cos(i*1D0)*cos(i*1D0)
  enddo

  ! Sum component wise and save result into vector c

  tBlock = dim3(256,1,1)
  grid = dim3(ceiling(real(n)/tBlock%x),1,1)

  a_d = a
  b_d = b

  call vadd<<<grid, tBlock>>>(a_d, b_d, c_d)

  c = c_d

  ! Sum up vector c and print result divided by n, this should equal 1 within error
  do i=1,n
     vsum = vsum +  c(i)
  enddo
  vsum = vsum/n
  print *, 'final result: ', vsum

end program main
