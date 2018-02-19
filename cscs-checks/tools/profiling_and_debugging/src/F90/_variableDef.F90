module VariableDef
    implicit none

    TYPE JacobiData
        integer :: iRows
        integer :: iCols
        integer :: iRowFirst
        integer :: iRowLast
        integer :: iIterMax
        double precision :: fAlpha
        double precision :: fRelax
        double precision :: fTolerance

!        /* calculated dx & dy */
        double precision :: fDx
        double precision :: fDy

!       /* pointers to the allocated memory */
        double precision, allocatable :: afU(:,:)
        double precision, allocatable :: afF(:,:)

!       /* start and end timestamps */
        double precision :: fTimeStart
        double precision :: fTimeStop

!       /* calculated residual (output jacobi) */
        double precision :: fResidual
!       /* effective interation count (output jacobi) */
        integer :: iIterCount

!       /* calculated error (output error_check) */
        double precision :: fError

!       /* MPI-Variables */
        integer :: iMyRank   !/* current process rank (number) */
        integer :: iNumProcs !/* how many processes */

   END TYPE JacobiData

end module VariableDef
