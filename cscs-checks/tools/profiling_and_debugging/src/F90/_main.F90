program MAIN
    !***********************************************************************
    ! program to solve a finite difference                                 *
    ! discretization of Helmholtz equation :                               *
    ! (d2/dx2)u + (d2/dy2)u - alpha u = f                                  *
    ! using Jacobi iterative method.                                       *
    !                                                                      *
    ! Modified: Abdelali Malih,    Aachen University (RWTH), 2007          *
    ! Modified: Sanjiv Shah,       Kuck and Associates, Inc. (KAI), 1998   *
    ! Author  : Joseph Robicheaux, Kuck and Associates, Inc. (KAI), 1998   *
    !                                                                      *
    ! Directives are used in this code to achieve paralleism.              *
    ! All do loops are parallized with default 'static' scheduling.        *
    !                                                                      *
    ! Input :  n - grid dimension in x direction                           *
    !          m - grid dimension in y direction                           *
    !          alpha - Helmholtz constant (always greater than 0.0)        *
    !          tol   - error tolerance for iterative solver                *
    !          relax - Successice over relaxation parameter                *
    !          mits  - Maximum iterations for iterative solver             *
    !                                                                      *
    ! On output                                                            *
    !       : u(n,m) - Dependent variable (solutions)                      *
    !       : f(n,m) - Right hand side function                            *
    !***********************************************************************

    use VariableDef
    use JacobiMod
    implicit none
    include 'mpif.h'

    TYPE(JacobiData) :: myData


!   sets default values or reads from stdin
!    * inits MPI and OpenMP if needed
!    * distribute MPI data, calculate MPI bounds
!    */
    call Init(mydata)

    if ( allocated(myData%afU) .and. allocated(myData%afF) ) then
!        /* matrix init */
        call InitializeMatrix(myData)

!        /* starting timer */
        mydata%fTimeStart = MPI_Wtime()

!        /* running calculations */
        call Jacobi(myData)

!        /* stopping timer */
        mydata%fTimeStop = MPI_Wtime()

!        /* error checking */
        call CheckError(myData)

!        /* print result summary */
        call PrintResults(myData)
    else
        write (*,*) " Memory allocation failed ...\n"
    end if

!    /* cleanup */
    call Finish(myData)

end program MAIN

subroutine Init (myData)
    use VariableDef
    implicit none
    include 'mpif.h'
    type(JacobiData), intent(inout) :: myData
    character(len=8) :: env = ' '
    integer :: ITERATIONS = 5
    integer :: provided
    integer :: version, subversion
    integer :: iErr, i
    integer :: omp_get_max_threads
    integer :: block_lengths(8), typelist(8), MPI_JacobiData
#if !defined(MPI_VERSION) || (MPI_VERSION>=2)
    integer (kind=MPI_ADDRESS_KIND) :: displacements(8), iStructDisp
#else
    integer :: displacements(8), iStructDisp
#endif
    ! MPI_Get_library_version:
    integer :: resultlen = -1
    character(len=MPI_MAX_LIBRARY_VERSION_STRING) :: mpilibversion

!    /* MPI Initialization */
#if !defined(MPI_VERSION) || (MPI_VERSION>=2)
    integer :: required = MPI_THREAD_FUNNELED
    call MPI_Init_thread(MPI_THREAD_FUNNELED, provided, iErr)
    if (iErr /= MPI_SUCCESS) then
        print*, "Abort: MPI_Init_thread unsuccessful"
        call MPI_Abort(MPI_COMM_WORLD, 38, iErr)
    else if (provided < required) then
        write (6,'(2(A,I1))') "Warning: MPI_Init_thread only provided level ", provided, "<", required
    endif
#else
    call MPI_Init(iErr)
    if (iErr /= MPI_SUCCESS) then
        print*, "Abort: MPI_Init unsuccessful"
        call MPI_Abort(MPI_COMM_WORLD, 38, iErr)
    endif
#endif

    call MPI_Comm_rank(MPI_COMM_WORLD, myData%iMyRank, iErr)
    call MPI_Comm_size(MPI_COMM_WORLD, myData%iNumProcs, iErr)
    if (myData%iMyRank == 0) then
        call get_environment_variable("ITERATIONS", env)
        if (len_trim(env) > 0) then
            read(env,*,iostat=iErr) i
            if ((iErr == 0) .and. (i > 0)) then
                ITERATIONS=i
            else
                print*, "Ignoring ITERATIONS=", env
            endif
        endif
        call MPI_Get_version(version, subversion, iErr)
        write (6,'(3(A,I1))') 'MPI-', version, '.', subversion, '#', provided
#ifdef _OPENMP
        write (6,*) 'Jacobi', myData%iNumProcs, 'MPI process(es) with', &
              omp_get_max_threads(), 'OpenMP-', _OPENMP, ' thread(s)/process'
#else
        write (6,*) 'Jacobi', myData%iNumProcs, 'MPI process(es) with', &
               ' 0 OpenMP-xxx thread(s)/process'
#endif

! MPI_Get_library_version:
        call MPI_Get_library_version(mpilibversion, resultlen, iErr)
        write (6,'(A)') trim(mpilibversion)
        flush(6)

!/* default medium */
        myData%iCols      = 2000
        myData%iRows      = 2000
        myData%fAlpha     = 0.8
        myData%fRelax     = 1.0
        myData%fTolerance = 1e-10
        myData%iIterMax   = ITERATIONS
#ifdef READ_INPUT
        write (*,*) 'Input n - matrix size in x direction: '
        read (5,*) myData%iCols
        write (*,*) 'Input m - matrix size in y direction: '
        read (5,*) myData%iRows
        write (*,*) 'Input alpha - Helmholts constant:'
        read (5,*) myData%fAlpha
        write (*,*) 'Input relax - Successive over-relaxation parameter:'
        read (5,*) myData%fRelax
        write (*,*) 'Input tol - error tolerance for iterative solver:'
        read (5,*) myData%fTolerance
        write (*,*) 'Input mits - Maximum iterations for solver:'
        read (5,*) myData%iIterMax
#elif defined DATA_LARGE
        myData%iCols      = 7000
        myData%iRows      = 7000
        myData%fAlpha     = 0.8
        myData%fRelax     = 1.0
        myData%fTolerance = 1e-12
        myData%iIterMax   = 2

#elif defined DATA_SMALL
        myData%iCols      = 200
        myData%iRows      = 200
        myData%fAlpha     = 0.8
        myData%fRelax     = 1.0
        myData%fTolerance = 1e-7
        myData%iIterMax   = 1000
#endif
        write (*,327) "-> matrix size: ", myData%iCols, myData%iRows
        write (*,329) "-> alpha: " , myData%fAlpha
        write (*,329) "-> relax: ", myData%fRelax
        write (*,329) "-> tolerance: ", myData%fTolerance
        write (*,328) "-> iterations: ", myData%iIterMax
        flush(6)
327     format (A22, I10, ' x ', I10)
328     format (A22, I10)
329     format (A22, F10.6)

    end if

!    /* Send input parameters to all procs */
    block_lengths = 1
    typelist(1) = MPI_INTEGER
    typelist(2) = MPI_INTEGER
    typelist(3) = MPI_INTEGER
    typelist(4) = MPI_INTEGER
    typelist(5) = MPI_INTEGER
    typelist(6) = MPI_DOUBLE_PRECISION
    typelist(7) = MPI_DOUBLE_PRECISION
    typelist(8) = MPI_DOUBLE_PRECISION
#if !defined(MPI_VERSION) || (MPI_VERSION>=2)
    call MPI_GET_ADDRESS(myData%iRows, displacements(1), iErr)
    call MPI_GET_ADDRESS(myData%iCols, displacements(2), iErr)
    call MPI_GET_ADDRESS(myData%iRowFirst, displacements(3), iErr)
    call MPI_GET_ADDRESS(myData%iRowLast, displacements(4), iErr)
    call MPI_GET_ADDRESS(myData%iIterMax, displacements(5), iErr)
    call MPI_GET_ADDRESS(myData%fAlpha, displacements(6), iErr)
    call MPI_GET_ADDRESS(myData%fRelax, displacements(7), iErr)
    call MPI_GET_ADDRESS(myData%fTolerance, displacements(8), iErr)
    call MPI_GET_ADDRESS(myData, iStructDisp, iErr)
#else
    call MPI_ADDRESS(myData%iRows, displacements(1), iErr)
    call MPI_ADDRESS(myData%iCols, displacements(2), iErr)
    call MPI_ADDRESS(myData%iRowFirst, displacements(3), iErr)
    call MPI_ADDRESS(myData%iRowLast, displacements(4), iErr)
    call MPI_ADDRESS(myData%iIterMax, displacements(5), iErr)
    call MPI_ADDRESS(myData%fAlpha, displacements(6), iErr)
    call MPI_ADDRESS(myData%fRelax, displacements(7), iErr)
    call MPI_ADDRESS(myData%fTolerance, displacements(8), iErr)
    call MPI_ADDRESS(myData, iStructDisp, iErr)
#endif

    displacements = displacements - iStructDisp

#if !defined(MPI_VERSION) || (MPI_VERSION>=2)
    call MPI_Type_create_struct(8, block_lengths, displacements, typelist, &
                                MPI_JacobiData, iErr)
#else
    call MPI_Type_struct(8, block_lengths, displacements, typelist, &
                                MPI_JacobiData, iErr)
#endif
    call MPI_Type_commit(MPI_JacobiData, iErr)

    call MPI_Bcast(myData, 1, MPI_JacobiData, 0, MPI_COMM_WORLD, iErr)

!    /* calculate bounds for the task working area */
    myData%iRowFirst = myData%iMyRank * (myData%iRows - 2) / myData%iNumProcs
    if (myData%iMyRank == myData%iNumProcs - 1) then
        myData%iRowLast = myData%iRows - 1
    else
        myData%iRowLast = (myData%iMyRank + 1) * (myData%iRows - 2) / myData%iNumProcs + 1
    end if

    allocate( myData%afU (0 : myData%iCols -1, myData%iRowFirst : myData%iRowLast))
    allocate( myData%afF (0 : myData%iCols -1, myData%iRowFirst : myData%iRowLast))

!    /* calculate dx and dy */
    myData%fDx = 2.0d0 / DBLE(myData%iCols - 1)
    myData%fDy = 2.0d0 / DBLE(myData%iRows - 1)

    myData%iIterCount = 0

end subroutine Init

subroutine InitializeMatrix (myData)
    !*********************************************************************
    ! Initializes data                                                   *
    ! Assumes exact solution is u(x,y) = (1-x^2)*(1-y^2)                 *
    !                                                                    *
    !*********************************************************************
    use VariableDef
    implicit none

    type(JacobiData), intent(inout) :: myData
    !.. Local Scalars ..
    integer :: i, j
    double precision :: xx, yy
    !.. Intrinsic Functions ..
    intrinsic DBLE

    ! Initilize initial condition and RHS

!$omp parallel do private (j, i, xx, yy)
    do j = myData%iRowFirst, myData%iRowLast
        do i = 0, myData%iCols -1
            xx = (-1.0 + myData%fDx*DBLE(i)) ! -1 < x < 1
            yy = (-1.0 + myData%fDy*DBLE(j)) ! -1 < y < 1
            myData%afU(i, j) = 0.0d0
            myData%afF(i, j) = - myData%fAlpha * (1.0d0 - DBLE(xx*xx))  &
                * (1.0d0 - DBLE(yy*yy)) - 2.0d0 * (1.0d0 - DBLE(xx*xx)) &
                - 2.0d0 * (1.0d0 - DBLE(yy*yy))
        end do
    end do
!$omp end parallel do
end subroutine InitializeMatrix

subroutine Finish(myData)
    use VariableDef
    implicit none

    integer :: iErr
    type(JacobiData), intent(inout) :: myData

    deallocate (myData%afU)
    deallocate (myData%afF)

    call MPI_Finalize(iErr)
end subroutine Finish

subroutine PrintResults(myData)
    use VariableDef
    implicit none

    type(JacobiData), intent(inout) :: myData

    if (myData%iMyRank == 0) then
        write (*,328) " Number of iterations : ", myData%iIterCount
        write (*,329) " Residual             : ", myData%fResidual
        write (*,329) " Solution Error       : ", myData%fError
        write (*,330) " Elapsed Time         : ", &
               myData%fTimeStop - myData%fTimeStart
        write (*,330) " MFlops/s             : ", &
               0.000013 * DBLE (myData%iIterCount) &
               * DBLE((myData%iCols - 2) * (myData%iRows - 2)) &
               / (myData%fTimeStop - myData%fTimeStart)
        write  (*,330) "SUCCESS"
        flush(6)
328     format (A, I8)
329     format (A, F15.12)
330     format (A, F15.6)
    end if
end subroutine PrintResults


subroutine CheckError(myData)
    use VariableDef
    implicit none
    include 'mpif.h'

    type(JacobiData), intent(inout) :: myData
    !.. Local Scalars ..
    integer :: i, j, iErr
    double precision :: error, temp, xx, yy
    !.. Intrinsic Functions ..
    intrinsic DBLE, SQRT
    ! ... Executable Statements ...
    error = 0.0d0

    do j = myData%iRowFirst, myData%iRowLast
        do i = 0, myData%iCols -1
            xx = -1.0d0 + myData%fDx * DBLE(i)
            yy = -1.0d0 + myData%fDy * DBLE(j)
            temp = myData%afU(i, j) - (1.0d0-xx*xx)*(1.0d0-yy*yy)
            error = error + temp*temp
        end do
    end do

    myData%fError = error
    call MPI_Reduce(myData%fError, error, 1, MPI_DOUBLE_PRECISION, MPI_SUM, 0, &
                    MPI_COMM_WORLD, iErr)
    myData%fError = sqrt(error) / DBLE(myData%iCols * myData%iRows)

end subroutine CheckError
