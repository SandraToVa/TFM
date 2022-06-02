PROGRAM PLOT_EMP
  !***********************************************************************
  !                            MAIN PROGRAM
  ! gfortran -g -fcheck=all -w emp_boot.f90
  !***********************************************************************

  IMPLICIT NONE
  INTEGER, PARAMETER :: nsc=29, nt=22
  INTEGER, PARAMETER :: nboot=30
  DOUBLE PRECISION, PARAMETER :: nsc_=1.D0/DBLE(nsc),nsc1_=1.D0/DBLE(nsc-1)
  DOUBLE PRECISION, PARAMETER :: nboot_=1.D0/DBLE(nboot), nbot_=1.D0/DBLE(nboot-1)
  DOUBLE PRECISION :: pmean(nt), pmeanboot(nboot,nt), boot, EMpoint(nboot,nt), blck(nsc,nt)
  DOUBLE PRECISION :: mean(nt), sigm(nt)
  DOUBLE PRECISION, allocatable :: x(:,:)
  INTEGER :: i, j, k, kt

  OPEN (10, file='prot_SP.dat')
  DO i=1,nsc
    READ(10,*) (blck(i,k),k=1,nt)		!Llegim les C_i(t) del fitxer de dades on columnes=k=t i files=i de 1 a N
  ENDDO
  CLOSE(10)

  DO k=1,nt
    pmean(k)=SUM(blck(:,k))*nsc_
  ENDDO

  ALLOCATE(x(nsc,nboot))
  call random_number(x)				!Numero aleatori de configuracions
  DO k=1,nt
    DO j=1,nboot
      boot =0.D0
      DO i=1,nsc				!suma de i on i=1,N i N=nsc es #de configuacions
        boot=boot+blck(int(x(i,j)*nsc+1),k)	!blck=C_rand(i)(t)
      ENDDO
      pmeanboot(j,k)=boot*nsc_			!Pas 2: pmeanboot(j,k)=C_b(t)
    ENDDO
  ENDDO

  kt=1
  EMpoint = 0.D0
  DO k=1,(nt-kt)							!k=t
    DO j=1,nboot							!j=b
        EMpoint(j,k)=log(pmeanboot(j,k)/pmeanboot(j,k+kt))/DBLE(kt)	!Pas 3: calcula E_b(t)
    ENDDO
  ENDDO
  DO i=1,(nt-kt)				!Suma per a cada temps i de b=1 asta N_b=nboot
    mean(i)=sum(EMpoint(:,i))*nboot_		!Pas 3: lo mean es la suma de E_b=EMpoint(i)
  ENDDO
  DO k=1,(nt-kt)
    sigm(k)=0.D0
    DO j=1,nboot
      sigm(k)=sigm(k)+(EMpoint(j,k)-mean(k))*(EMpoint(j,k)-mean(k))
    ENDDO
    sigm(k)=dsqrt(sigm(k)*nboot_*DBLE(nsc)*nsc1_)
  ENDDO
  OPEN (10, file="EMP_prot_boot.dat")
    DO k=1,(nt-kt)
      WRITE(10,*) k,mean(k),sigm(k)		!mean(k)=\barr(E)(t), 
    ENDDO
  CLOSE(10)

!Per calcular la matriu de cov i les chi2
  OPEN (11, file="EMP_prot_boot_var.dat")
    WRITE(11,*) nsc,nboot,nt 
  CLOSE(11)

  OPEN (12, file="EMP_prot_boot_param.dat")
    DO k=1,(nt-kt)
      WRITE(12,*) EMpoint(:,k)			!Escribim les e_b(t)	 
    ENDDO					!Les files son t, fila 1 es t=1 asta 21
						!Les columnes son b, columna 1 es b=1 asta 30
  CLOSE(12)

END PROGRAM PLOT_EMP
