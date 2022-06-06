#!/usr/bin/python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
import math
import itertools
import matplotlib.patches as patches

#mpl.rcParams['text.usetex'] = True
#mpl.rcParams['text.latex.preview'] = True
from matplotlib.backends.backend_pdf import PdfPages
#Per al ajust
from scipy.optimize import curve_fit
from scipy.optimize import minimize



with open('EMP_prot_boot.dat', 'r') as f:
    data = f.read()

data = data.split('\n')[:-1]
xboot = [float(row.split()[0]) for row in data]
yboot = [float(row.split()[1]) for row in data]
eboot = [float(row.split()[2]) for row in data]

with open('EMP_prot_jack.dat', 'r') as f:
    data = f.read()

data = data.split('\n')[:-1]
xjack = [float(row.split()[0])+0.1 for row in data]
yjack = [float(row.split()[1]) for row in data]
ejack = [float(row.split()[2]) for row in data]

#Per calulcar chi2 i cov matriu

nsc=29
nt=22
nboot=30


with open('EMP_prot_boot_param.dat', 'r') as f:
    data = f.read()

data = data.split('\n')[:-1]
#Files lo t comensant en t=1 i acabant en 21. Columnes b. [t=1[b],t=2[b],...]
E_b=np.array([[float(i) for i in row.split()] for row in data])
#Los primers index son 0

t=[i for i in range(1,nt+1)]
b=[i for i in range(1,nboot+1)]

#Calulem la matriu de covariancia per a tots los t
#t i t' van de =[1,21] matriu de 21x21
cov=[[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]

for l in range(0,nt-1): #l files de cov, t
    for c in range(0,nt-1): #c columnes de cov, t'
        suma=0
        for p in range(0,nboot): #sumatori de les b per acada element de la matriu t,t'
            suma=suma+(E_b[l][p]-yboot[l])*(E_b[c][p]-yboot[c])
        suma=(nsc/(nsc+1))*(1/nboot)*suma
        cov[l].append(suma)
cov=np.array(cov)

cov_=np.linalg.inv(cov)     #inversa de la matriu covariant

#Faig un loop en diferents temps

#L'ajust y=b te sentit en x>7 si x<7 cal afegir una exponencial negativa e^-En·x (pq hi ha contaminació de estats excitats). Per tant a temps llargs domina E_0 estat de energia més baix
#Difrents proves: y=b i anar variant lo rang per veure com varia la qualitat del ajust el valor b que seria la massa E_0 del sistema; y=b+e^ax
#Fem que tot sige los mateixos intervals, de 5 fins 17 en intervals de mida 5 minim

#AgAfo i faig un fit y=b en un interval x1-x2 me dona un valor central i un valor estadistic. Vario lo interval i cada fit de estos dona central i estadistic. A banda podem posar-li una exponencial. La diferencia més gran entre el valor central de estos fits i lo meu escollit es lo error sistematic.


counter_i=0
#chi square
chi2_l=[[],[],[],[],[],[],[]] #lineal
chi2_e=[[],[],[],[],[],[],[]]
#valor central
central_l=[[],[],[],[],[],[],[]]
central_e=[[],[],[],[],[],[],[]]
#error estadistic
sigma_l=[[],[],[],[],[],[],[]]
sigma_e=[[],[],[],[],[],[],[]]
#error total = estad + sistem suma quadràtica
error_t_l=[[],[],[],[],[],[],[]]
error_t_e=[[],[],[],[],[],[],[]]

for i in range(5,12):       #Temps inicial del fit
    #Lo valor minim del interval es 5
    j=[j for j in range(i+5,17)]

    counter_f=0
    for f in j:              #Temps finals possibles

        #Ajust lineal
        def func_l(t,d):
            return d

        #Ajust exponencial
        def func_e(t, a,b,c):
            return a * np.exp(-b * t) + c

        #Data del fit
        X=[]
        Y=[]
        print(i)
        print(f)
        X=np.array([float(x) for x in range(i,f+1)])
        Y=np.array([float(yboot[x-1]) for x in range(i,f+1)])


        #Fit
        popt_l, pcov_l = curve_fit(func_l, X, Y, p0=[1.19], maxfev=10000)
        popt_e, pcov_e = curve_fit(func_e, X, Y, p0=[0,1,1.2], maxfev=10000, bounds=([0.,0.,1.],[1.,20,3.]))

        yfit_l=func_l(X, *popt_l)
        yfit_e=func_e(X, *popt_e)


        chi_l=0
        chi_e=0
        n=0
        for x in range(i,f+1):
            m=0
            for y in range(i,f+1):  #numero de x del yboot i eboot
            #per a yfit cal fer un index diferent pq ja comensa de la x corresponent -> n
                chi_l=chi_l+(yboot[x-1]-yfit_l)*cov_[x-1][y-1]*(yboot[y-1]-yfit_l)
                chi_e=chi_e+(yboot[x-1]-yfit_e[n])*cov_[x-1][y-1]*(yboot[y-1]-yfit_e[m])
                m+=1
            n+=1


        #Ajust del fit minimitzant la chi2 ymin=c
        cmin_l=[] #Llista en les c que per cada b dona la chib minima
        cmin_e=[]
        sigma_estad_l=0
        sigma_estad_e=0
        for b in range(0,nboot): #Llista de les chi2 minimes per a cada b

            def fun_chib_l(c):
                chib_l=0
                for x in range(i,f+1):
                    for y in range(i,f+1):
                        #Pas 6
                        chib_l=chib_l+(E_b[x-1][b]-func_l(x-1,c))*cov_[x-1][y-1]*(E_b[y-1][b]-func_l(y-1,c))
                return chib_l

            def fun_chib_e(c):
                chib_e=0
                def func_e2(t, c):
                    return c[0] * np.exp(-c[1] * t) + c[2]

                for x in range(i,f+1):
                    for y in range(i,f+1):
                        #Pas 6
                        chib_e=chib_e+(E_b[x-1][b]-func_e2(x-1,c))*cov_[x-1][y-1]*(E_b[y-1][b]-func_e2(y-1,c))
                return chib_e

            c0=[1.95] #First guess de la c

            x0=[0.,1.,1.95] #First guesses de la a,b,c
            bnds=((0.,1.),(0.,20.),(1.,3.))#Mateixos bounds que usats antes

            #Lineal
            res_l=minimize(fun_chib_l,c0,method='Nelder-Mead',tol=1e-6)
            c_m_l=res_l.x[0]
            cmin_l.append(c_m_l)
            #Exponencial
            res_e=minimize(fun_chib_e,x0,method='Nelder-Mead',bounds=bnds,tol=1e-6)
            c_m_e=res_e.x[2]
            cmin_e.append(c_m_e)


        #Ara per a fer lo pas 6, ordenem la llista de cmin i calculem lo 5/6 i 1/6 quartil
        #lineal
        cmin_l.sort()   #ordeno la llista de petit a gran
        cmin_l=np.array(cmin_l)
        cmin_l=[elemento - popt_l[0] for elemento in cmin_l]    #llista de les c-\bar{c}
        q_5=0
        q_1=0

        q_5=np.quantile(cmin_l,5./6)
        q_1=np.quantile(cmin_l,1./6)

        #Calculem la sigma
        sigma_estad_l=(q_5-q_1)/2

        #exponencial
        cmin_e.sort()
        cmin_e=np.array(cmin_e)
        cmin_e=[abs(elemento - popt_e[2]) for elemento in cmin_e]
        q_5=0
        q_1=0

        q_5=np.quantile(cmin_e,5./6)
        q_1=np.quantile(cmin_e,1./6)


        #Calculem la sigma
        sigma_estad_e=(q_5-q_1)/2

        #Afegim a les matrius lo valor de chi square, lo valor central de l'ajust=la constant i el valor estadistic
        chi2_l[counter_i].append(chi_l)
        central_l[counter_i].append(popt_l[0])
        sigma_l[counter_i].append(sigma_estad_l)
        chi2_e[counter_i].append(chi_e)
        central_e[counter_i].append(popt_e[2])
        sigma_e[counter_i].append(sigma_estad_e)

        #L'error sistematic lo caluclo al final pero aqui lo poso per a poder GRAFICAR
        sigma_sist_l=0.006717993289348412
        sigma_sist_e=0.19952003648979955
        #L'eror total es
        sigma_t_l=math.sqrt(sigma_sist_l**2+sigma_estad_l**2)
        sigma_t_e=math.sqrt(sigma_sist_e**2+sigma_estad_e**2)

        error_t_l[counter_i].append(sigma_t_l)
        error_t_e[counter_i].append(sigma_t_e)

        #Per fer el plot dels ajustos
        xplot=np.linspace(i,f,num=(f-i)*100)
        yplot_l=[]
        for num in range(0,(f-i)*100):
            yplot_l.append(func_l(xplot, *popt_l))
        yplot_e=func_e(xplot, *popt_e)

        plt.rc('text', usetex=True)
        plt.rc('font', family='serif', size='12')

        fig = plt.figure(figsize=(8,6))

        plt.subplots_adjust(left=0.08, bottom=0.08, right=0.98, top=0.95, wspace=0.21, hspace=0.2)

        #########################
        #No cal fer un gràfic cada vegada -> Cal canviar-ho
        #PLOT LINEAL
        fig1 = fig.add_subplot(1,1,1)

        fig1.set_title("Effective mass plot")
        fig1.set_ylabel(r'$\mathrm{m} \,\mathrm{(l.u.)}$')
        fig1.set_xlabel(r'$t \,\mathrm{(l.u.)}$')
        fig1.set_ylim([1.10,1.3])
        fig1.set_xlim([0,20.5])
        #plt.xticks([5,10,15,20])
        plt.minorticks_on()
        fig1.axes.tick_params(which='both',direction='in')
        fig1.yaxis.set_ticks_position('both')
        fig1.xaxis.set_ticks_position('both')
        fig1.errorbar(xboot,yboot, yerr=eboot, c='#ED553B', ls='None', marker='o', markersize=6, capsize=1, elinewidth=0.7,label="Bootstrap")
        fig1.errorbar(xjack,yjack, yerr=ejack, c='#20639B', ls='None', marker='o', markersize=6, capsize=1, elinewidth=0.7,label="Jackknive")
        #Plot del ajust
        plt.plot(xplot, yplot_l, 'r-', label='fit: c=%5.3f' % tuple(popt_l))
        #Error de l'ajust
        fig1.add_patch(
            patches.Rectangle(
                (i, popt_l[0]-sigma_t_l), #Esquina inferior izquierda
                f-i,                        #Ancho
                2*sigma_t_l,
                edgecolor = 'white',
                facecolor = '#ffcccc',
                fill=True
                ) )
        plt.legend()
        #plt.show()
        with PdfPages(str(counter_i) + str(counter_f) + 'EMP_prot_lineal.pdf') as pdf:
            pdf.savefig(fig)

        plt.close('all')

        #PLOT EXPONENCIAL
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif', size='12')

        fig = plt.figure(figsize=(8,6))

        plt.subplots_adjust(left=0.08, bottom=0.08, right=0.98, top=0.95, wspace=0.21, hspace=0.2)
        fig1 = fig.add_subplot(1,1,1)

        fig1.set_title("Effective mass plot")
        fig1.set_ylabel(r'$\mathrm{m} \,\mathrm{(l.u.)}$')
        fig1.set_xlabel(r'$t \,\mathrm{(l.u.)}$')
        fig1.set_ylim([1.10,1.3])
        fig1.set_xlim([0,20.5])
        #plt.xticks([5,10,15,20])
        plt.minorticks_on()
        fig1.axes.tick_params(which='both',direction='in')
        fig1.yaxis.set_ticks_position('both')
        fig1.xaxis.set_ticks_position('both')
        fig1.errorbar(xboot,yboot, yerr=eboot, c='#ED553B', ls='None', marker='o', markersize=6, capsize=1, elinewidth=0.7,label="Bootstrap")
        fig1.errorbar(xjack,yjack, yerr=ejack, c='#20639B', ls='None', marker='o', markersize=6, capsize=1, elinewidth=0.7,label="Jackknive")
        #Plot del ajust
        plt.plot(xplot, yplot_e, 'r-', label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt_e))
        #Error de l'ajust

        plt.legend()
        #plt.show()
        with PdfPages(str(counter_i) + str(counter_f) + 'EMP_prot_exp.pdf') as pdf:
            pdf.savefig(fig)

        plt.close('all')



        counter_f += 1
    counter_i += 1

print('LINEAL###############################')
print('* chi2 =',chi2_l)
print('* central =',central_l)
print('* error estadistic =',sigma_l)

#Millor resultat
#Lo ajust lineal 4,0 te la chi2 més baixa en 10.95: en lo fit 6 0
#Per al error sistematic
#Lo calcul de l'error sistematic es algo que faig ara al final despres de haver fet tota la resta. Agafo el millor fit: 40 que dona valor central[3][0]=1.1995200365039278
#Restem aquest numero en tots los elements de la llista central
flat_central=itertools.chain(*central_l) #Fem que central sigue una sola llista
flat_central=list(flat_central)
sistematic=[abs(elemento - central_l[6][0]) for elemento in flat_central]
#Error sistematic es lo maxim error de la llista
sist_l=max(sistematic)
print('* error sistemàtic =',sist_l)
print('* error total =',error_t_l)

print('EXPONENCIAL###############################')
print('* chi2 =',chi2_e)
print('* central =',central_e)
print('* error estadistic =',sigma_e)

#Millor resultat
#Lo ajust lineal 4,0 te la chi2 més baixa en 10.95: 4 0 for 11 and 16 we get a= [0.] b= 1.19952
#Per al error sistematic
#Lo calcul de l'error sistematic es algo que faig ara al final despres de haver fet tota la resta. Agafo el millor fit: 40 que dona valor central[3][0]=1.1995200365039278
#Restem aquest numero en tots los elements de la llista central
flat_central=itertools.chain(*central_e) #Fem que central sigue una sola llista
flat_central=list(flat_central)
sistematic=[abs(elemento - central_e[6][0]) for elemento in flat_central]
#Error sistematic es lo maxim error de la llista
sist_e=max(sistematic)
print('* error sistemàtic =',sist_e)
print('* error total =',error_t_e)
