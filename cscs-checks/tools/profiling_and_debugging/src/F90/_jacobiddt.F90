module JacobiMod
    use VariableDef
    implicit none

    contains

    subroutine Jacobi(myData)
        implicit none
        include 'mpif.h'
        !********************************************************************
        ! Subroutine HelmholtzJ                                             *
        ! Solves poisson equation on rectangular grid assuming :            *
        ! (1) Uniform discretization in each direction, and                 *
        ! (2) Dirichlect boundary conditions                                *
        ! Jacobi method is used in this routine                             *
        !                                                                   *
        ! Input : n,m   Number of grid points in the X/Y directions         *
        !         dx,dy Grid spacing in the X/Y directions                  *
        !         alpha Helmholtz eqn. coefficient                          *
        !         omega Relaxation factor                                   *
        !         myData%afF(n,m) Right hand side function                  *
        !         myData%afU(n,m) Dependent variable/Solution               *
        !         tol    Tolerance for iterative solver                     *
        !         maxit  Maximum number of iterations                       *
        !                                                                   *
        ! Output : myData%afU(n,m) - Solution                               *
        !********************************************************************

        !.. Formal Arguments ..
        type(JacobiData), intent(inout) :: myData

        !.. Local Scalars ..
        integer :: i, j, iErr
        double precision :: ax, ay, b, residual, fLRes, tmpResd

        !.. Local Arrays ..
        double precision, allocatable :: uold(:,:)

        !.. Intrinsic Functions ..
        intrinsic DBLE, SQRT

        allocate(uold (0 : myData%iCols -1, myData%iRowFirst : myData%iRowLast))

        ! ... Executable Statements ...
        ! Initialize coefficients

        if (allocated(uold)) then
            ax = 1.0d0 / (myData%fDx * myData%fDx)      ! X-direction coef
            ay = 1.0d0 / (myData%fDx * myData%fDx)      ! Y-direction coef
            b = -2.0d0 * (ax + ay) - myData%fAlpha      ! Central coeff
            residual = 10.0d0 * myData%fTolerance

            do while (myData%iIterCount < myData%iIterMax .and. residual > myData%fTolerance)
                residual = 0.0d0

            ! Copy new solution into old
                call ExchangeJacobiMpiData(myData, uold)
!$omp parallel private(fLRes, tmpResd, i)
!$omp do reduction(+:residual)
                  ! Compute stencil, residual, & update
                   do j = myData%iRowFirst + 1, myData%iRowLast - 1
                       do i = 1, myData%iCols - 2
                           ! Evaluate residual
                           fLRes = (ax * (uold(i-1, j) + uold(i+1, j)) &
                                  + ay * (uold(i, j-1) + uold(i, j+1)) &
                                  + b * uold(i, j) - myData%afF(i, j)) / b

                           ! Update solution
                           myData%afU(i, j) = uold(i, j) - myData%fRelax * fLRes

                           ! Accumulate residual error
                           residual = residual + fLRes * fLRes
                       end do
                   end do
!$omp end do
!$omp end parallel
                  tmpResd = residual
                  call MPI_Allreduce(tmpResd, residual, 1, MPI_DOUBLE_PRECISION, &
                                MPI_SUM, MPI_COMM_WORLD, Ierr)

                 ! Error check
                 myData%iIterCount = myData%iIterCount + 1
                 residual = SQRT(residual) / DBLE(myData%iCols * myData%iRows)

            ! End iteration loop
            end do
            myData%fResidual = residual
            deallocate(uold)
        else
           write (*,*) 'Error: cant allocate memory'
           call Finish(myData)
           stop
        end if
    end subroutine Jacobi

    subroutine ExchangeJacobiMpiData (myData, uold)
    !    use VariableDef
        implicit none
        include 'mpif.h'
        type(JacobiData), intent(inout) :: myData
        double precision, intent(inout) :: uold(0: myData%iCols -1, myData%iRowFirst : myData%iRowLast)
        integer :: request(4), status(MPI_STATUS_SIZE, 4)
        integer, parameter :: iTagMoveLeft = 10, iTagMoveRight = 11
        integer i, j, iErr, iReqCnt

        iReqCnt = 0
        if (myData%iMyRank /= 0) then
!!           /*  receive stripe mlo from left neighbour blocking */
            iReqCnt = iReqCnt + 1
            call MPI_Irecv( uold(0, myData%iRowFirst), myData%iCols, &
                           MPI_DOUBLE_PRECISION, myData%iMyRank - 1, &
                           iTagMoveRight, MPI_COMM_WORLD, request(iReqCnt), iErr)
        end if

        if (myData%iMyRank /= myData%iNumProcs - 1) then
!!           /* receive stripe mhi from right neighbour blocking */
            iReqCnt = iReqCnt + 1
            call MPI_Irecv( uold(0, myData%iRowLast), myData%iCols,  &
                           MPI_DOUBLE_PRECISION, myData%iMyRank + 1, &
                           iTagMoveLeft, MPI_COMM_WORLD, request(iReqCnt), iErr)
        end if

        if (myData%iMyRank /= myData%iNumProcs - 1) then
!!           /* send stripe mhi-1 to right neighbour async */
            iReqCnt = iReqCnt + 1
            call MPI_Isend( myData%afU(0, myData%iRowLast - 1), myData%iCols, &
                            MPI_DOUBLE_PRECISION, myData%iMyRank + 1,         &
                            iTagMoveRight, MPI_COMM_WORLD, request(iReqCnt), iErr)
        end if

        if (myData%iMyRank /= 0) then
!!           /* send stripe mlo+1 to left neighbour async */
            iReqCnt = iReqCnt + 1
            call MPI_Isend( myData%afU(0, myData%iRowFirst + 1), myData%iCols, &
                           MPI_DOUBLE_PRECISION, myData%iMyRank - 1,           &
                           iTagMoveLeft, MPI_COMM_WORLD, request(iReqCnt), iErr)
        end if
!$omp parallel do
        do j = myData%iRowFirst + 1, myData%iRowLast - 1
            do i = 0, myData%iCols - 1
                uold(i, j) = myData%afU(i, j)
            end do
        end do
!$omp end parallel do

        call MPI_Waitall(iReqCnt, request, status, iErr)
    end subroutine ExchangeJacobiMpiData
end module JacobiMod
