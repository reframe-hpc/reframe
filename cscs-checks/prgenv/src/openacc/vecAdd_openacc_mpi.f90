      program main
      include 'mpif.h'
 
      ! Size of vectors
      integer :: n = 100000
  
      ! Input vectors
      real(8),dimension(:),allocatable :: a
      real(8),dimension(:),allocatable :: b  
      ! Output vector
      real(8),dimension(:),allocatable :: c
   
      integer :: i
      real(8) :: sum

      call MPI_Init(ierr)
      call MPI_Comm_size(MPI_COMM_WORLD, isize, ierr)
      call MPI_Comm_rank(MPI_COMM_WORLD, irank, ierr)
   
      ! Allocate memory for each vector
      allocate(a(n))
      allocate(b(n))
      allocate(c(n))
   
      ! Initialize content of input vectors, vector a[i] = sin(i)^2 vector b[i] = cos(i)^2
      do i=1,n
          a(i) = sin(i*1D0)*sin(i*1D0)
          b(i) = cos(i*1D0)*cos(i*1D0)  
      enddo
   
      ! Sum component wise and save result into vector c
   
      !$acc kernels copyin(a(1:n),b(1:n)), copyout(c(1:n))
      do i=1,n
          c(i) = a(i) + b(i)
      enddo
      !$acc end kernels
   
      sum = 0d0
      ! Sum up vector c and print result divided by n, this should equal 1 within error
      do i=1,n
          sum = sum +  c(i)
      enddo
      sum = sum/n/isize

      if (irank.eq.0) then
          call MPI_Reduce(MPI_IN_PLACE, sum, 1, MPI_REAL8, MPI_SUM, 0, MPI_COMM_WORLD, ierr)
          print *, 'final result: ', sum
      else
          call MPI_Reduce(sum, sum, 1, MPI_REAL8, MPI_SUM, 0, MPI_COMM_WORLD, ierr)
      end if
   
      ! Release memory
      deallocate(a)
      deallocate(b)
      deallocate(c)

      call MPI_Finalize(ierr)
  
      end program
