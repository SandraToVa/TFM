# TFM
Codis usats dunarnt el TFM del màster de física nuclear.
- 'emp_boot.f90': genera els arxius 'EMP_prot_boot.dat' i 'EMP_prot_boot_param.dat'.  Usa 'prot_SP.dat'.
- 'emp_jack.f90': genera els arxius 'EMP_prot_jack.dat'. (falta acabar) Usa 'prot_SP.dat'.
- 'plot_fortran_sandra.py': codi que fa un ajust lineal i exponencial de *manera conjunta* de les dades de massa efectiva obtingudes amb bootstrap and jackknife. Usa els arxius 'EMP_prot_boot.dat', 'EMP_prot_jack.dat' i 'EMP_prot_boot_param.dat'. (falta la part de jack)
- 'plot_fortran_sandra2.py': backup del codi
- Capeta 'Ajustos': pdf cretas amb els ajutos i els errors per a cada interval de temps creats amb 'plot_fortran_sandra.py'
