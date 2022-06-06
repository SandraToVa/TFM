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
#Per al ajust y=b
from sklearn import linear_model
#Per al ajust y=a*e(-b*x)+c
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

##Primer Ajust entre varios temps (massa simple)
#gboot = (yboot[6]+yboot[15])/2
##Calcul de la chi/dof
#chiboot = 0
#for i in range(6,15):
#    chiboot = chiboot + ((yboot[i]-gboot)**2)/yboot[i]
#dof = 15 - 6
#chidofboot = chiboot/dof
#print('chi/dof for boot =', chidofboot)
#print('mass entre 7.5 i 16 =', gboot)


#Faig un loop en diferents temps

#L'ajust y=b te sentit en x>7 si x<7 cal afegir una exponencial negativa e^-En·x (pq hi ha contaminació de estats excitats). Per tant a temps llargs domina E_0 estat de energia més baix
#Difrents proves: y=b i anar variant lo rang per veure com varia la qualitat del ajust el valor b que seria la massa E_0 del sistema; y=b+e^ax
#Agafar lo fit que dona millor chi2 
#Error sistematic = diferencia entre el resultat bo (valor central del resultat) - valor de cada fit adicional que vas fent

#Agfo i faig un fit y=b en un interval x1-x2 me dona un valor central i un valor estadistic. Vario lo interval i cada fit de estos dona central i estadistic. A banda podem posar-li una exponencial. La diferencia més gran entre el valor central de estos fits i lo meu escollit es lo error sistematic.


print('AJUST LINEAL----------------')
counter_i=0
chi2=[[],[],[],[],[]]
central=[[],[],[],[],[]]
#Llista de llistes de chi2_b
chi2_b=[[],[],[],[],[]]
sigma=[[],[],[],[],[]]
error_t=[[],[],[],[],[]]

for i in range(7,12):       #Temps inicial del fit
    #Lo valor minim del interval es 5
    j=[j for j in range(i+5,17)]

    counter_f=0
    for f in j:              #Temps finals possibles
        #Per a usar los valors de i i f com a x1 i x2 cal fer float(i), float(j)
        #Ajust regressió lineal usant linealregresion de py
        lr = linear_model.Lasso(alpha=1)
        X=np.array([[x] for x in range(i,f+1)])
        Y=np.array(tuple(yboot[x-1] for x in range(i,f+1)))
        
        
        lr.fit(X, Y)
        yfit = lr.predict(X)
        
        #Valor de la pendiente o coeficiente "a" i Valor de la intersección o coeficiente "b"
 #       print(counter_i, counter_f,'for', i, 'and', f, 'we get', 'a=',lr.coef_, 'b=',lr.intercept_)
        #Variables de suport per al calcul de chi square, valor central i estadistic de cada ajust
        
        
        chi=0
        chi_b=[]
        n=0
        for x in range(i,f+1):
            m=0
            for y in range(i,f+1):  #numero de x del yboot i eboot
            #per a yfit cal fer un index diferent pq ja comensa de la x corresponent -> n
                chi=chi+(yboot[x-1]-yfit[n])*cov_[x-1][y-1]*(yboot[y-1]-yfit[m])
                m+=1
            n+=1
        
            
        #Ajust del fit minimitzant la chi2 ymin=c
        cmin=[] #Llista en les c que per cada b dona la chib minima
        sigma_c=0
        for b in range(0,nboot): #Llista de les chi2 minimes per a cada b
            chimin=[]
            chibc=0
            
            def fun_chib(c):
                chib=0
                for x in range(i,f+1):
                    for y in range(i,f+1):
                        #Pas 6
                        chib=chib+(E_b[x-1][b]-c)*cov_[x-1][y-1]*(E_b[y-1][b]-c)
                return chib
            
            c0=[0.195] #First guess de la c
            
            res=minimize(fun_chib,c0,method='Nelder-Mead',tol=1e-6)
            c_m=res.x[0]
            cmin.append(c_m)
                        
            
        #Ara per a fer lo pas 6, ordenem la llista de cmin i calculem lo 5/6 i 1/6 quartil
        cmin.sort()
        cmin=np.array(cmin)
        cmin=[elemento - lr.intercept_ for elemento in cmin]
        q_5=0
        q_1=0
        
        q_5=np.quantile(cmin,5./6)
        q_1=np.quantile(cmin,1./6)

            
        #Calculem la sigma
        sigma_c=(q_5-q_1)/2
        
        #Afegim a les matrius lo valor de chi square, lo valor central de l'ajust=la constant i el valor estadistic=sqrt((sum_1^2(yboot-yfit)^2)/(n-2))
        chi2[counter_i].append(chi)
        central[counter_i].append(lr.intercept_)
        chi2_b[counter_i].append(chi_b)
        sigma[counter_i].append(sigma_c)
        
        #L'error sistematic lo caluclo al final pero aqui lo poso per a poder fitejarlo
        sigma_sist=0.006466904353966463
        sigma_estad=sigma_c
        #L'eror total es
        sigma_t=math.sqrt(sigma_sist**2+sigma_estad**2)
        
        error_t[counter_i].append(sigma_t)
        
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif', size='12')

        fig = plt.figure(figsize=(8,6))

        plt.subplots_adjust(left=0.08, bottom=0.08, right=0.98, top=0.95, wspace=0.21, hspace=0.2)

        #########################
        #No cal fer un gràfic cada vegada -> Cal canviar-ho
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
        fig1.plot(X, yfit, 'r-', label='fit: b=%5.3f' % lr.intercept_)
        #Error de l'ajust
        fig1.add_patch(
            patches.Rectangle(
                (i, lr.intercept_-sigma_t), #Esquina inferior izquierda
                f-i,                        #Ancho
                2*sigma_t,
                edgecolor = 'white',
                facecolor = '#ffcccc',
                fill=True
                ) )
        plt.legend()
        #plt.show()
        with PdfPages(str(counter_i) + str(counter_f) + 'EMP_prot_lineal.pdf') as pdf:
            pdf.savefig(fig)
            
        plt.close('all')
        
        counter_f += 1
    counter_i += 1


print('* chi2 y=b,=',chi2)
print('* central y=b,=',central)
print('* error estadistic y=b,=',sigma)

#Millor resultat
#Lo ajust lineal 4,0 te la chi2 més baixa en 10.95: 4 0 for 11 and 16 we get a= [0.] b= 1.19952
#Per al error sistematic 
#Lo calcul de l'error sistematic es algo que faig ara al final despres de haver fet tota la resta. Agafo el millor fit: 40 que dona valor central[3][0]=1.1995200365039278
#Restem aquest numero en tots los elements de la llista central
flat_central=itertools.chain(*central) #Fem que central sigue una sola llista
flat_central=list(flat_central)
sistematic=[abs(elemento - central[4][0]) for elemento in flat_central]
#Error sistematic es lo maxim error de la llista
sist=max(sistematic)
print('* error sistemàtic y=b,=',sist)
print('* error total per a y=b, =',error_t)

#Despues quan lo ajust exponencial vaigue be, he de fer lo error sisteàtic entre los fits lineal i los exponencial junts


#################################################################################################

print('AJUST EXPONENCIAL----------------')
#Exactament tot igual pero en y=a*e^-bx+c

counter_i=0
chi2=[[],[],[],[],[]]
central=[[],[],[],[],[]]
estadistic=[[],[],[],[],[]]
#En ajust exponencial comencem de més pronte
for i in range(3,8):       #Temps inicial del fit
    #Lo valor minim del interval es 5
    j=[j for j in range(i+5,17)]

    counter_f=0
    for f in j:              #Temps finals possibles
        #Per a usar los valors de i i f com a x1 i x2 cal fer float(i), float(j)
        #Ajust exponencial
        def func(t, a, b, c):
            return a * np.exp(-b * t) + c
        
        #Data del fit
        X=np.array([float(x) for x in range(i,f+1)])
        Y=np.array([float(yboot[x]) for x in range(i,f+1)])
        
        #Fit
        popt, pcov = curve_fit(func, X, Y, p0=[0,1,1.2], maxfev=10000, bounds=([0,-10,-10],[1.,10,10]))
        
                               
        yfit=func(X, *popt)
        
        #Valor de la pendiente o coeficiente "a" i Valor de la intersección o coeficiente "b"
 #       print(counter_i, counter_f,'for', i, 'and', f, 'we get', popt)
        #Variables de suport per al calcul de chi square, valor central i estadistic de cada ajust
        chi=0
        chib=0
        estad=0
        n=0
        for x in range(i,f+1):
            m=0
            for y in range(i,f+1):  #numero de x del yboot i eboot
            #per a yfit cal fer un index diferent pq ja comensa de la x corresponent -> n
                chi=chi+(yboot[x-1]-yfit[n])*cov_[x-1][y-1]*(yboot[y-1]-yfit[m])
                #Pas 6
                m+=1
            n+=1
            
            
        #Ajust del fit minimitzant la chi2 ymin=c
        cmin=[] #Llista en les c que per cada b dona la chib minima
        sigma_c=0
        for b in range(0,nboot): #Llista de les chi2 minimes per a cada b
            chimin=[]
            chibc=0
            
            def fun_chib(c):
                def func(t, c):
                    return c[0] * np.exp(-c[1] * t) + c[2]  #C=a,b,c
                
                chib=0
                for x in range(i,f+1):
                    for y in range(i,f+1):
                        #Pas 6
                        chib=chib+(E_b[x-1][b]-func(x-1,c))*cov_[x-1][y-1]*(E_b[y-1][b]-func(y-1,c))
                return chib
            
            x0=[0.,1.,0.195] #First guess de la a,b,c
            bnds=((0.,1.),(-10.,10.),(-10.,10.))#Mateixos bounds que usats antes
            
            res=minimize(fun_chib,x0,method='Nelder-Mead',bounds=bnds,tol=1e-6)
            c_m=res.x[2]
            cmin.append(c_m)
            
        #Ara per a fer lo pas 6, ordenem la llista de cmin i calculem lo 5/6 i 1/6 quartil
        cmin.sort()
        cmin=np.array(cmin)
        cmin=[abs(elemento - popt[2]) for elemento in cmin]
        q_5=0
        q_1=0
        
        q_5=np.quantile(cmin,5./6)
        q_1=np.quantile(cmin,1./6)

            
        #Calculem la sigma
        sigma_c=(q_5-q_1)/2
        
        #Afegim a les matrius lo valor de chi square, lo valor central de l'ajust=la constant i el valor estadistic=sqrt((sum_1^2(yboot-yfit)^2)/(n-2))
        chi2[counter_i].append(chi)
        central[counter_i].append(popt[2])
        sigma[counter_i].append(sigma_c)
        
        #L'error sistematic lo caluclo al final pero aqui lo poso per a poder fitejarlo
        sigma_sist=0.9926249652096809       #Crec q dona molt gran! No se ploteja dins del gran dona un error molt gran!
        sigma_estad=sigma_c
        #L'eror total es
        sigma_t=math.sqrt(sigma_sist**2+sigma_estad**2)
        
        error_t[counter_i].append(sigma_t)
        
        
        #Vull que el plot sigui smooth, per tant he de calcular la yfit per a molts de valors de x
        xplot=np.linspace(i,f,num=(f-i)*100)
        yplot=func(xplot, *popt)
        
        #Per a plotejar los errors
        popt_s=[popt[0],popt[1],popt[2]+sigma_t]
        popt_i=[popt[0],popt[1],popt[2]-sigma_t]
        
        yplot_sup=func(xplot, *popt_s)
        yplot_inf=func(xplot, *popt_i)
        
        
        
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif', size='12')

        fig = plt.figure(figsize=(8,6))

        plt.subplots_adjust(left=0.08, bottom=0.08, right=0.98, top=0.95, wspace=0.21, hspace=0.2)

        #########################
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
        plt.plot(xplot, yplot, 'r-', label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))
        plt.plot(xplot, yplot_sup)
        plt.plot(xplot, yplot_inf)
        plt.legend()
        #plt.show()
        with PdfPages(str(counter_i) + str(counter_f) + 'EMP_prot_exp.pdf') as pdf:
            pdf.savefig(fig)
            
        plt.close('all')
        
        counter_f += 1
    counter_i += 1


print('* chi2 y=b,=',chi2)
print('* central y=b,=',central)
print('* error estadistic y=b,=',sigma)


#Per al error sistematic 
#Lo calcul de l'error sistematic es algo que faig ara al final despres de haver fet tota la resta. Agafo el millor fit: 4 1
#Restem aquest numero en tots los elements de la llista central
flat_central=itertools.chain(*central) #Fem que central sigue una sola llista
flat_central=list(flat_central)
sistematic=[abs(elemento - central[4][1]) for elemento in flat_central]
#Error sistematic es lo maxim error de la llista
sist=max(sistematic)
print('* error sistemàtic y=b,=',sist)
print('* error total per a y=b, =',error_t)
##################################################################################################


