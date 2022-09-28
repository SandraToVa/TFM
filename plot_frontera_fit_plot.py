import plot_frontera_fit as pf
import sys
#!/usr/bin/python
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
import math
import itertools
#import random #MI: millor fes servir np.random.normal
import matplotlib.patches as patches
from matplotlib.patches import Polygon

#mpl.rcParams['text.usetex'] = True
#mpl.rcParams['text.latex.preview'] = True
from matplotlib.backends.backend_pdf import PdfPages

import h5py
#Per al ajust
from scipy.optimize import curve_fit
from scipy.optimize import minimize

nsc=454  #N # Utilitza nsc=454 si fas servir les dades del fitxer h5
nt=19
nboot=nsc    #Nb
nsc_=1./nsc
nsc1_=1./(nsc-1)
nboot_=1./nboot
nbot_=1./(nboot-1)
kt=1

dir = 'Op_no5_' + sys.argv[1] +'/'

color=['tab:blue','tab:orange','tab:green','tab:red']
lightc=['aliceblue','linen','honeydew','mistyrose']
darkc=['lightblue','moccasin','lightgreen','lightcoral']

counter_i=0
#chi square
chi2_l=[[],[],[]] #lineal
chi2_e=[[],[],[]]
#valor central
central_l=[[],[],[]]
central_e=[[],[],[]]
#error estadistic
sigma_l=[[],[],[]]
sigma_e=[[],[],[]]
#Llita en los fits de cada ajust
fit_l=[[],[],[]]
fit_e=[[],[],[]]

millor_chi2_l=100.   #per saber quin es el millor ajust
millor_chi2_e=100.

xboot=list(range(1, nt))
yboot=pf.yboot[int(sys.argv[1])]
eboot=pf.eboot[int(sys.argv[1])]
E_b=pf.E_b[int(sys.argv[1])]


#Ajust lineal
def func_l(t,d):
    return d

#Ajust exponencial
def func_e(t, a,b,c):
    return a * np.exp(-b * t) + c

#Matriu de covariància per a tots els Temps=cov_t
cov_t=np.zeros((nt-1,nt-1))
for l in range(0,nt-1): #l files de cov, t
    for c in range(0,nt-1): #c columnes de cov, t'
        suma=0
        for p in range(0,nboot): #sumatori de les b per acada element de la matriu t,t'
            suma=suma+(E_b[l][p]-yboot[l])*(E_b[c][p]-yboot[c])
        suma=(nsc/(nsc-1))*(1/nboot)*suma
        cov_t[l][c]=suma
cov_t=np.array(cov_t)

for i in range(7,10):       #Temps inicial del fit (7,10) o (3,12)
    #Lo valor minim del interval es 5
    counter_f=0
    for f in range(i+5,15):              #Temps finals possibles (15) o (17)

        #Mida de l'interval
        j=f-i+1
        #Matriu de covariància per a cada interval de Temps=cov
        cov=[]
        cov=cov_t[i-1:f,i-1:f]
        cov_=np.linalg.inv(cov)     #inversa de la matriu covariant
        #Trobem lo millor fit minimitzant la chi2
        #Com la matriu cov ja esta feta per a aquest interval de temps corresponent, creo dos contadors: n,m
        def fun_chi_l(c):
            chi_l=0
            n=0
            for x in range(i,f+1):
                m=0
                for y in range(i,f+1):
                    #Pas 6
                    chi_l=chi_l+(yboot[x-1]-func_l(x-1,c))*cov_[n][m]*(yboot[y-1]-func_l(y-1,c))
                    m+=1
                n+=1
            return chi_l

        def fun_chi_e(c):
            chi_e=0
            def func_e2(t, c):
                return c[0] * np.exp(-c[1] * t) + c[2]
            n=0
            for x in range(i,f+1):
                m=0
                for y in range(i,f+1):
                    #Pas 6
                    chi_e=chi_e+(yboot[x-1]-func_e2(x-1,c))*cov_[n][m]*(yboot[y-1]-func_e2(y-1,c))
                    m+=1
                n+=1
            return chi_e

        c0=[1.95] #First guess de la c

        x0=[0.,1.,1.95] #First guesses de la a,b,c
        bnds=((0.,1.),(-20.,20.),(-3.,3.))#Mateixos bounds que usats antes

        #Lineal
        res_l=minimize(fun_chi_l,c0,method='Nelder-Mead',tol=1e-6)
        c_l=res_l.x
        central_l[counter_i].append(c_l[0])
        fit_l[counter_i].append(c_l)
        #Exponencial
        res_e=minimize(fun_chi_e,x0,method='Nelder-Mead',bounds=bnds,tol=1e-6)
        c_e=res_e.x
        central_e[counter_i].append(c_e[2])
        fit_e[counter_i].append(c_e)

        #En aixó hem trobat los valors de les c i c_llista que minimitzen la chi2
        #Ara calculem la chi en estos valors
        #Graus de llibertat=(sample size=j=mida del interval) - parametres
        dof_l=j-1
        dof_e=j-3
        chi_l=fun_chi_l(res_l.x[0])/dof_l
        chi2_l[counter_i].append(chi_l)
        chi_e=fun_chi_e(res_e.x)/dof_l
        chi2_e[counter_i].append(chi_e)


        #Ajust del fit minimitzant la chi2 ymin=c
        cmin_l=[] #Llista en les c que per cada b dona la chib minima
        cmin_e=[] #Llista per a a b i c ja que ho usare per a trobar els quantiles
        amin_e=[]
        bmin_e=[]
        sigma_estad_l=0
        sigma_estad_e=0
        for b in range(0,nboot): #Llista de les chi2 minimes per a cada b

            def fun_chib_l(c):
                chib_l=0
                n=0
                for x in range(i,f+1):
                    m=0
                    for y in range(i,f+1):
                        #Pas 6
                        chib_l=chib_l+(E_b[x-1][b]-func_l(x-1,c))*cov_[n][m]*(E_b[y-1][b]-func_l(y-1,c))
                        m+=1
                    n+=1
                return chib_l

            def fun_chib_e(c):
                chib_e=0
                def func_e2(t, c):
                    return c[0] * np.exp(-c[1] * t) + c[2]
                n=0
                for x in range(i,f+1):
                    m=0
                    for y in range(i,f+1):
                        #Pas 6
                        chib_e=chib_e+(E_b[x-1][b]-func_e2(x-1,c))*cov_[n][m]*(E_b[y-1][b]-func_e2(y-1,c))
                        m+=1
                    n+=1
                return chib_e

            c0=[1.95] #First guess de la c

            x0=[0.,1.,1.95] #First guesses de la a,b,c
            bnds=((0.,1.),(-20.,20.),(-3.,3.))#Mateixos bounds que usats antes

            #Lineal
            res_l=minimize(fun_chib_l,c0,method='Nelder-Mead',tol=1e-6)
            c_m_l=res_l.x[0]
            cmin_l.append(c_m_l)
            #Exponencial
            res_e=minimize(fun_chib_e,x0,method='Nelder-Mead',bounds=bnds,tol=1e-6)
            c_m_e=res_e.x[2]
            cmin_e.append(c_m_e)
            amin_e.append(res_e.x[0])
            bmin_e.append(res_e.x[1])


        #Ara per a fer lo pas 6, ordenem la llista de cmin i calculem lo 5/6 i 1/6 quartil
        #lineal
        cmin_l.sort()   #ordeno la llista de petit a gran
        cmin_l=np.array(cmin_l)
        cmin_l=[elemento - c_l[0] for elemento in cmin_l]    #llista de les c-\bar{c}
        q_5=0
        q_1=0

        q_5=np.quantile(cmin_l,5./6)
        q_1=np.quantile(cmin_l,1./6)

        #Calculem la sigma
        sigma_estad_l=(q_5-q_1)/2

        #exponencial
        #Per a l'exponencial cal fer los quantiles de a b i class: a+delta_a, b+delta_b i c+delta_c
        #c_e es la llista que conte a b i c
        #delta_e contindrà les deltes
        delta_e=[]
        cmin_e.sort()
        amin_e.sort()
        bmin_e.sort()
        cmin_e=np.array(cmin_e)
        amin_e=np.array(amin_e)
        bmin_e=np.array(bmin_e)
        #a i delta_a
        amin_e=[elemento - c_e[0] for elemento in amin_e]
        q_5=0
        q_1=0
        q_5=np.quantile(amin_e,5./6)
        q_1=np.quantile(amin_e,1./6)

        delta_a=(q_5-q_1)/2
        delta_e.append(delta_a)
        #b i delta_b
        bmin_e=[elemento - c_e[1] for elemento in bmin_e]
        q_5=0
        q_1=0
        q_5=np.quantile(bmin_e,5./6)
        q_1=np.quantile(bmin_e,1./6)

        delta_b=(q_5-q_1)/2
        delta_e.append(delta_b)
        #c i delta_c
        cmin_e=[elemento - c_e[2] for elemento in cmin_e]
        q_5=0
        q_1=0
        q_5=np.quantile(cmin_e,5./6)
        q_1=np.quantile(cmin_e,1./6)

        delta_c=(q_5-q_1)/2
        delta_e.append(delta_c)
        #Genero 1000 numeros aleatoris en distribu gaussiana on mu=a,b,c i sigma=delta_a,b,c
        f_sup=[]    #llista dels valors per a cada t del quantile 5/6
        f_inf=[]    # """" 1/6
        t_errors=np.linspace(i,f,f-i)

        # MI: calcular primer els valors aleatoris
        a=np.random.normal(c_e[0],delta_e[0],1000)
        b=np.random.normal(c_e[1],delta_e[1],1000)
        c=np.random.normal(c_e[2],delta_e[2],1000)
        for t in range(i,f):
            f_t=[]  #f_t=llista de el valor de la funció usant los 1000 numeros aleatoris
            for n in range(0,1000):
                f_t.append(func_e(t,a[n],b[n],c[n]))

            f_t.sort()
            f_t=np.array(f_t)                #ordeno la Llista

            q_5=np.quantile(f_t,5./6)
            q_1=np.quantile(f_t,1./6)
            f_sup.append(q_5)
            f_inf.append(q_1)

        #Calculem la sigma estadistica = delta_c
        sigma_estad_e=delta_e[2]

        #Afegim a les matrius el valor estadistic
        sigma_l[counter_i].append(sigma_estad_l)
        sigma_e[counter_i].append(sigma_estad_e)

        #Per fer el plot dels ajustos
        xplot=np.linspace(i,f,num=(f-i)*100)
        yplot_l=[]
        for num in range(0,(f-i)*100):
            yplot_l.append(func_l(xplot, c_l[0]))
        yplot_e=func_e(xplot, c_e[0],c_e[1],c_e[2])


        plt.rc('text', usetex=True)
        plt.rc('font', family='serif', size='12')

        fig = plt.figure(figsize=(8,6))

        plt.subplots_adjust(left=0.08, bottom=0.08, right=0.98, top=0.95, wspace=0.21, hspace=0.2)

        #########################            #No cal fer un gràfic cada vegada pero m'ajuda a visualitzar l'ajust -> Cal canviar-ho quan tot vagi be
            #PLOT LINEAL
        fig1 = fig.add_subplot(1,1,1)
        fig1.set_title("Effective mass plot")
        fig1.set_ylabel(r'$\mathrm{m} \,\mathrm{(l.u.)}$')
        fig1.set_xlabel(r'$t \,\mathrm{(l.u.)}$')
        fig1.set_xlim([0,19])
        #plt.xticks([5,10,15,20])
        plt.minorticks_on()
        fig1.axes.tick_params(which='both',direction='in')
        fig1.yaxis.set_ticks_position('both')
        fig1.xaxis.set_ticks_position('both')
        fig1.errorbar(xboot,yboot, yerr=eboot, c=color[int(sys.argv[1])], ls='None', marker='o', markersize=6, capsize=1, elinewidth=0.7,label={"Operator"+str(3+int(sys.argv[1]))})
        ##fig1.errorbar(xjack,yjack, yerr=ejack, c='#20639B', ls='None', marker='o', markersize=6, capsize=1, elinewidth=0.7,label="Jackknive")
        #Plot del ajust
        plt.plot(xplot, yplot_l, c=color[int(sys.argv[1])], ls='-', label='fit: c=%5.3f l.u.' % tuple(c_l))
        #Error de l'ajust (només l'estadistic)
        fig1.add_patch(
            patches.Rectangle(
                (i, c_l[0]-sigma_estad_l), #Esquina inferior izquierda
                f-i,                        #Ancho
                2*sigma_estad_l,
                edgecolor = 'white',
                facecolor = lightc[int(sys.argv[1])],
                fill=True
                ) )
        plt.legend()
        #plt.show()
        with PdfPages(dir+str(counter_i) + str(counter_f) + 'Op' + sys.argv[1] +'_lineal.pdf') as pdf:
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
        fig1.set_xlim([0,19])
        #plt.xticks([5,10,15,20])
        plt.minorticks_on()
        fig1.axes.tick_params(which='both',direction='in')
        fig1.yaxis.set_ticks_position('both')
        fig1.xaxis.set_ticks_position('both')
        fig1.errorbar(xboot,yboot, yerr=eboot, c=color[int(sys.argv[1])], ls='None', marker='o', markersize=6, capsize=1, elinewidth=0.7,label={"Operator"+str(3+int(sys.argv[1]))})
        ##fig1.errorbar(xjack,yjack, yerr=ejack, c='#20639B', ls='None', marker='o', markersize=6, capsize=1, elinewidth=0.7,label="Jackknive")
        #Plot del ajust
        plt.plot(xplot, yplot_e, c=color[int(sys.argv[1])], ls='-', label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(c_e))
        #Error de l'ajust (només l'estadistic)
        plt.fill_between(t_errors, f_sup, f_inf, color=lightc[int(sys.argv[1])]) #MI: el representa la banda, en comptes de les dues linies

        plt.legend()
        #plt.show()
        with PdfPages(dir+str(counter_i) + str(counter_f) + 'Op' + sys.argv[1] +'_exp.pdf') as pdf:
            pdf.savefig(fig)

        plt.close('all')

        #Aguardo les dades del millor fit
        #millor_c_l, milllor_c_e, millor_f_sup, millor_f_inf, millor_sigma_estad_l, millor_sigma_estad_e
        if millor_chi2_l>chi_l:
            millor_i_l=i
            millor_f_l=f
            millor_c_l=c_l
            millor_sigma_estad_l=sigma_estad_l
            millor_chi2_l=chi_l

        if millor_chi2_e>chi_e:
            millor_i_e=i
            millor_f_e=f
            millor_c_e=c_e
            millor_sigma_estad_e=sigma_estad_e
            millor_f_sup=f_sup
            millor_f_inf=f_inf
            millor_chi2_e=chi_e

        print(counter_i,counter_f)
        counter_f += 1
    counter_i += 1



print('LINEAL###############################')
print('* chi2 =',chi2_l)
print('* central =',central_l)
print('* fit =',fit_l)  #Comprovo que es lo mateix q el central
print('* error estadistic =',sigma_l)

#Millor resultat
#Lo ajust lineal 0,1 te la chi2 més baixa en 7.263617072777405
#Per al error sistematic
#Lo calcul de l'error sistematic es algo que faig ara al final despres de haver fet tota la resta. Agafo el millor fit: 40 que dona valor central[3][0]=1.1995200365039278
#Restem aquest numero en tots los elements de la llista central
flat_central=itertools.chain(*central_l) #Fem que central sigue una sola llista
flat_central=list(flat_central)
sistematic=[elemento - millor_c_l[0] for elemento in flat_central]
#Error sistematic es lo maxim error de la llista
sist_l=max(sistematic)
print('* error sistemàtic =',sist_l)
#L'eror total es
sigma_t_l=math.sqrt(sist_l**2+millor_sigma_estad_l**2)
print('* millor error total =',sigma_t_l)
#MILLOR PLOT LINEAL
#Per fer el plot dels ajustos
xplot=np.linspace(millor_i_l,millor_f_l,num=(millor_f_l-millor_i_l)*100)
yplot_l=[]
for num in range(0,(millor_f_l-millor_i_l)*100):
    yplot_l.append(func_l(xplot, millor_c_l[0]))


plt.rc('text', usetex=True)
plt.rc('font', family='serif', size='17')

fig = plt.figure(figsize=(8,6))

plt.subplots_adjust(left=0.13, bottom=0.13, right=0.98, top=0.95, wspace=0.21, hspace=0.2)
fig1 = fig.add_subplot(1,1,1)

fig1.set_title("Effective mass plot")
fig1.set_ylabel(r'$\mathrm{E} \,\mathrm{(l.u.)}$')
fig1.set_xlabel(r'$t \,\mathrm{(l.u.)}$')
fig1.set_xlim([0,19])
#plt.xticks([5,10,15,20])
plt.minorticks_on()
fig1.axes.tick_params(which='both',direction='in')
fig1.yaxis.set_ticks_position('both')
fig1.xaxis.set_ticks_position('both')
fig1.errorbar(xboot,yboot, yerr=eboot, c=color[int(sys.argv[1])], ls='None', marker='o', markersize=6, capsize=1, elinewidth=0.7)
##fig1.errorbar(xjack,yjack, yerr=ejack, c='#20639B', ls='None', marker='o', markersize=6, capsize=1, elinewidth=0.7,label="Jackknive")
#Plot del ajust
plt.plot(xplot, yplot_l, c=color[int(sys.argv[1])], ls='-', label='fit: k=%5.3f (l.u.)' % tuple(millor_c_l))
#Error de l'ajust (estad+sistematic)
fig1.add_patch(
    patches.Rectangle(
        (millor_i_l, millor_c_l[0]-sigma_t_l), #Esquina inferior izquierda
        millor_f_l-millor_i_l,                        #Ancho
        2*sigma_t_l,
        linewidth=0,
        facecolor = darkc[int(sys.argv[1])],
        fill=True,
        label='Total error'
        ) )
#Error només estadistic
fig1.add_patch(
    patches.Rectangle(
        (millor_i_l, millor_c_l[0]-millor_sigma_estad_l), #Esquina inferior izquierda
        millor_f_l-millor_i_l,                        #Ancho
        2*millor_sigma_estad_l,
        linewidth=0,
        facecolor = lightc[int(sys.argv[1])],
        fill=True,
        label='Statistical error'
        ) )
plt.legend()
#plt.show()
with PdfPages(dir+'best' + 'Op' + sys.argv[1] + '_lineal_no5.pdf') as pdf:
    pdf.savefig(fig)

plt.close('all')

print('EXPONENCIAL###############################')
print('* chi2 =',chi2_e)
print('* central =',central_e)
print('* fit =',fit_e)
print('* error estadistic =',sigma_e)
print(millor_chi2_e)

#Millor resultat
#Lo ajust lineal 6,0 te la chi2 més baixa en 7.987054040388158
#Per al error sistematic
#Lo calcul de l'error sistematic es algo que faig ara al final despres de haver fet tota la resta. Agafo el millor fit: 40 que dona valor central[3][0]=1.1995200365039278
#Restem aquest numero en tots los elements de la llista central
flat_central=itertools.chain(*central_e) #Fem que central sigue una sola llista
flat_central=list(flat_central)
sistematic=[elemento-millor_c_e[2] for elemento in flat_central]
#Error sistematic es lo maxim error de la llista
sist_e=max(sistematic)
print('* error sistemàtic =',sist_e)
sigma_t_e=math.sqrt(sist_e**2+millor_sigma_estad_e**2)
print('* millor error total =',sigma_t_e)

#Per fer el plot dels ajustos
#MILLOR PLOT EXPONENCIAL
xplot=np.linspace(millor_i_e,millor_f_e,num=(millor_f_e-millor_i_e)*100)
yplot_e=func_e(xplot, millor_c_e[0],millor_c_e[1],millor_c_e[2])
t_errors=np.linspace(millor_i_e,millor_f_e,millor_f_e-millor_i_e)

plt.rc('text', usetex=True)
plt.rc('font', family='serif', size='17')

fig = plt.figure(figsize=(8,6))

plt.subplots_adjust(left=0.13, bottom=0.13, right=0.98, top=0.95, wspace=0.21, hspace=0.2)
fig1 = fig.add_subplot(1,1,1)

fig1.set_title("Effective mass plot")
fig1.set_ylabel(r'$\mathrm{E} \,\mathrm{(l.u.)}$')
fig1.set_xlabel(r'$t \,\mathrm{(l.u.)}$')
fig1.set_xlim([0,19])
#plt.xticks([5,10,15,20])
plt.minorticks_on()
fig1.axes.tick_params(which='both',direction='in')
fig1.yaxis.set_ticks_position('both')
fig1.xaxis.set_ticks_position('both')
fig1.errorbar(xboot,yboot, yerr=eboot, c=color[int(sys.argv[1])], ls='None', marker='o', markersize=6, capsize=1, elinewidth=0.7)
##fig1.errorbar(xjack,yjack, yerr=ejack, c='#20639B', ls='None', marker='o', markersize=6, capsize=1, elinewidth=0.7,label="Jackknive")
#Plot del ajust
plt.plot(xplot, yplot_e, c=color[int(sys.argv[1])], ls='-', label='fit: a=%5.3f, b=%5.3f, c=%5.3f (l.u.)' % tuple(millor_c_e))
#Error de l'ajust
fig1.add_patch(
    patches.Rectangle(
        (millor_i_e, millor_c_e[2]-sigma_t_e), #Esquina inferior izquierda
        millor_f_e-millor_i_e,                        #Ancho
        2*sigma_t_e,
        linewidth=0,
        facecolor = darkc[int(sys.argv[1])],
        fill=True,
        label='Total error'
        ) )
#Error només estadistic
fig1.add_patch(
    patches.Rectangle(
        (millor_i_e, millor_c_e[2]-millor_sigma_estad_e), #Esquina inferior izquierda
        millor_f_e-millor_i_e,                        #Ancho
        2*millor_sigma_estad_e,
        linewidth=0,
        facecolor = lightc[int(sys.argv[1])],
        fill=True,
        label='Statistical error'
        ) )
#plt.fill_between(t_errors, sigma_t_e_sup, sigma_t_e_inf, color='#ffa6a6', label='Total error', alpha=1) #MI: el representa la banda, en comptes de les dues linies


plt.legend()
#plt.show()
with PdfPages(dir+'best'+ 'Op' + sys.argv[1] +'_exp_no5.pdf') as pdf:
    pdf.savefig(fig)

plt.close('all')
