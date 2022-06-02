PROGRAM PLOT_EMP
  !***********************************************************************
  !                            MAIN PROGRAM
  ! gfortran -g -fcheck=all -w emp_jack.f90
  !***********************************************************************

  IMPLICIT NONE
  INTEGER, PARAMETER :: nsc=29, nt=22
  DOUBLE PRECISION, PARAMETER :: nsc_=1.D0/DBLE(nsc),nsc1_=1.D0/DBLE(nsc-1)
  DOUBLE PRECISION :: pmean(nt), pmeanjk(nsc,nt), EMpoint(nsc,nt), blck(nsc,nt)
  DOUBLE PRECISION :: mean(nt), sigm(nt)
  INTEGER :: i, j, k, kt

  OPEN (10, file='prot_SP.dat')
  DO i=1,nsc
    READ(10,*) (blck(i,k),k=1,nt)
  ENDDO
  CLOSE(10)

  DO k=1,nt
    pmean(k)=SUM(blck(:,k))*nsc_
  ENDDO

  DO k=1,nt
    DO j=1,nsc
      pmeanjk(j,k)=(pmean(k)*DBLE(nsc)-blck(j,k))*nsc1_
    ENDDO
  ENDDO

  kt=1
  EMpoint = 0.D0
  DO k=1,(nt-kt)
    DO j=1,nsc
        EMpoint(j,k)=log(pmeanjk(j,k)/pmeanjk(j,k+kt))/DBLE(kt)
    ENDDO
  ENDDO

  DO i=1,(nt-kt)
    mean(i)=sum(EMpoint(1:nsc,i))*nsc_
  ENDDO

  DO k=1,(nt-kt)
    sigm(k)=0.D0
    DO j=1,nsc
      sigm(k)=sigm(k)+(EMpoint(j,k)-mean(k))*(EMpoint(j,k)-mean(k))
    ENDDO
    sigm(k)=dsqrt(sigm(k)*nsc_*DBLE(nsc-1))
  ENDDO

  OPEN (10, file="EMP_prot_jack.dat")
    DO k=1,(nt-kt)
      WRITE(10,*) k,mean(k),sigm(k)
    ENDDO
  CLOSE(10)

END PROGRAM PLOT_EMP
