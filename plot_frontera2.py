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

nsc=454  #N # Utilitza nsc=454 si fas servir les dades del fitxer h5
nt=19
nboot=nsc    #Nb
nsc_=1./nsc
nsc1_=1./(nsc-1)
nboot_=1./nboot
nbot_=1./(nboot-1)
#Creo un bucle per a plotejer totes les dades juntes
#Numero de operadors n_op
n_op=11  #10
yboot=[]
eboot=[]
for op in range(0,n_op):
    #1. Llegim les dades Ci(t)
    #DELS FITXER .H5
    fh5 = h5py.File('\\Users\\Sandra\\Documents\\GitHub\\TFM\\qblocks_matrix_irreps_cl3_32_48_b6p1_m0p2450_frontera-002.h5', 'r')
    blck = 0.5*(np.real(np.array(fh5['B2_I1_A1_f'][0:nsc,op,0,op,1,0:nt])+np.real(np.array(fh5['B2_I1_A1_b'][0:nsc,op,0,op,1,0:nt]))))

    pmean=np.zeros(nt)
    for k in range(0,nt):
        suma=0
        for i in range(0,nsc):
            suma=suma+blck[i][k]
        pmean[k]=suma*nsc_

    #2. Creem les Cb(t)
    x=np.random.uniform(size=(nsc,nboot))  #Matriu de num aleatoris entre 0 i 1

    pmeanboot=np.zeros((nboot,nt))
    for k in range(0,nt):
        for j in range(0,nboot):
            boot=0.
            for i in range(0,nsc):
                boot=boot+blck[int(x[i][j]*nsc)][k]
            pmeanboot[j][k]=boot*nsc_   #Ara hem generat les Nb bootstrap samples Cb(t)

    #3. Calculem Eb(t)
    kt=1
    EMpoint=0.
    EMpoint=np.zeros((nboot,(nt-kt)))
    for k in range(0,(nt-kt)):
        for j in range(0,nboot):
            EMpoint[j][k]=np.log(pmeanboot[j][k]/pmeanboot[j][k+kt])/kt


    #Clculem \Bar{E}(t) i errors
    mean=np.zeros(nt-kt)
    for k in range(0,(nt-kt)):
        suma=0
        for j in range(0,nboot):
            suma=suma+EMpoint[j][k]
        mean[k]=suma*nboot_
    sigm=np.zeros(nt-kt)
    for k in range(0,(nt-kt)):
        sigma=0
        for j in range(0,nboot):
            sigma=sigma+(EMpoint[j][k]-mean[k])*(EMpoint[j][k]-mean[k])
        sigm[k]=np.sqrt(sigma*nboot_*nsc*nsc1_)
    #Dades
    yboot.append(mean)
    eboot.append(sigm)

    print(op)


xboot=list(range(1, nt))

plt.rc('text', usetex=True)
plt.rc('font', family='serif', size='12')

fig = plt.figure(figsize=(8,6))

plt.subplots_adjust(left=0.08, bottom=0.08, right=0.98, top=0.95, wspace=0.21, hspace=0.2)
##################            #No cal fer un gràfic cada vegada pero m'ajuda a visualitzar l'ajust -> Cal canviar-ho quan tot vagi be
#PLOT LINEAL
fig1 = fig.add_subplot(1,1,1)
fig1.set_title("Effective mass plot")
fig1.set_ylabel(r'$\mathrm{m} \,\mathrm{(l.u.)}$')
fig1.set_xlabel(r'$t \,\mathrm{(l.u.)}$')
fig1.set_ylim([2.10,2.6]) #1.10,1.3
fig1.set_xlim([0,22]) #0,20.5
#plt.xticks([5,10,15,20])
plt.minorticks_on()
fig1.axes.tick_params(which='both',direction='in')
fig1.yaxis.set_ticks_position('both')
fig1.xaxis.set_ticks_position('both')
#Plots dels diferents operadors
for i in range(0,n_op):
    fig1.errorbar(xboot,yboot[i], yerr=eboot[i], ls='None', marker='o', markersize=6, capsize=1, elinewidth=0.7, label=("operador"+str(i+1)))
plt.legend()
#plt.show()
with PdfPages('B2_I1_A1_plot.pdf') as pdf:
    pdf.savefig(fig)

plt.close('all')
