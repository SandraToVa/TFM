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

nsc=327
nt=18
nboot=nsc    #Nb
nsc_=1./nsc
nsc1_=1./(nsc-1)
nboot_=1./nboot
nbot_=1./(nboot-1)
#Creo un bucle per a plotejer totes les dades juntes
#Numero de operadors n_op
n_oplist = [3,11,19,27,35] #Barions sense moment
#n_oplist = [3,11,19,27,35,4,12,20,28,36] #Barions sense moment, amb moment n=1
#n_oplist = [3,11,19,27,35,4,12,20,28,36,5,13,21,29,37] #Barions sense moment, amb moment n=1, n=2
#n_oplist = [3,11,19,27,35,4,12,20,28,36,5,13,21,29,37,6,14,22,30,38] #Barions sense moment, amb moment n=1, n=2, n=3
n_op = len(n_oplist)
n_smear = 1
ndim = n_op*n_smear
yboot=[]
eboot=[]
#op=3,11
#DELS FITXER .H5
#fh5 = h5py.File('\\Users\\Sandra\\Documents\\GitHub\\TFM\\qblocks_matrix_irreps_cl3_32_48_b6p1_m0p2450_frontera-002.h5', 'r')
fh5 = h5py.File('/Users/marcilla/My Drive (marcilla@uw.edu)/NPLQCD/Hdib-variational/variational-autofiiter/var_data/qblocks_strange_matrix_irreps_cl3_32_48_b6p1_m0p2450_andes.h5', 'r')
blckB1 = np.zeros((nsc,nt))
blckB1 = 0.5*(np.real(np.array(fh5['B1_G1_f'][0:nsc,0,0,0,0,0:nt])+np.real(np.array(fh5['B1_G1_b'][0:nsc,0,0,0,0,0:nt]))))
blckB2 = np.zeros((nsc,ndim,ndim,nt))
for i,opsrc in enumerate(n_oplist):
    for j,opsnk in enumerate(n_oplist):
        for ii in range(n_smear):
            for jj in range(n_smear):
                iii = i+ii*n_op
                jjj = j+jj*n_op
                blckB2[:,iii,jjj,:] = 0.5*(np.real(np.array(fh5['B2_I1_A1_f'][0:nsc,opsrc,ii,opsnk,jj,0:nt])+np.real(np.array(fh5['B2_I1_A1_b'][0:nsc,opsrc,ii,opsnk,jj,0:nt]))))
                
coefsinglet = [np.sqrt(1/8), np.sqrt(3/8)*np.sqrt(1/3), np.sqrt(3/8)*np.sqrt(2/3), np.sqrt(1/2)*np.sqrt(1/2), np.sqrt(1/2)*np.sqrt(1/2)]
coefsingletmat = np.zeros((5,5))
for i in range(5):
    for j in range(5):
        coefsingletmat[i,j]=coefsinglet[i]*coefsinglet[j]

ndimirrep = int(ndim/5)
blckB2irrep=np.zeros((nsc,ndimirrep,ndimirrep,nt))
for i in range(ndimirrep):
    for j in range(ndimirrep):
        for ii in range(5):
            for jj in range(5):
                blckB2irrep[:,i,j,:] += coefsingletmat[ii,jj]*blckB2[:,ii+i*5,jj+j*5,:]

blck2=np.zeros((nsc,ndimirrep,ndimirrep,nt))
for op_src in range(ndimirrep):
    for op_snk in range(ndimirrep):
        #Var1 pre-processing antes de diagonalitzar C_ab(t)
        for k in range(nt):
            blck2[:,op_src,op_snk,k]=0.5*(blckB2irrep[:,op_src,op_snk,k]+blckB2irrep[:,op_snk,op_src,k])/np.sqrt(blckB2irrep[:,op_src,op_src,0]*blckB2irrep[:,op_snk,op_snk,0])

#Var1.2 Les matrius C_b(t)=pmeanboot correlator samples bootstrap samples NxN=[opsrc][opsnk]
#Millor passar array de op especifics
pmeanbootB1=np.zeros((nboot,nt))
pmeanbootB2=np.zeros((nboot,ndimirrep,ndimirrep,nt))
x=np.random.uniform(size=(nsc,nboot))  #Matriu de num aleatoris entre 0 i 1
for j in range(nboot):
    bootB1=np.zeros((nt))
    bootB2=np.zeros((ndimirrep,ndimirrep,nt))
    for i in range(nsc):
        bootB1=bootB1+blckB1[int(x[i][j]*nsc),:]
        bootB2=bootB2+blck2[int(x[i][j]*nsc),:,:,:]
    pmeanbootB1[j,:]=bootB1*nsc_
    pmeanbootB2[j,:,:,:]=bootB2*nsc_   #Ara hem generat les Nb bootstrap samples Cb(t)
    

if ndimirrep < 2:
    C_diag = pmeanbootB2
else:
    #Var3.a find the cholesky decomposition of pmeanboot2(t0)
    t_0=4-1 #t0=4
    #Tot per t_ref=2*t_0=8
    t_ref=2*(t_0+1)-1   #El -1 es per la llista q comensa en 0
    C_diag=np.zeros((nboot,ndimirrep,ndimirrep,nt))
    for n in range(nboot):
        L=np.linalg.cholesky(pmeanbootB2[n,:,:,t_0])
        L_=np.linalg.inv(L) #Matriu inversa
        Lt=L.T   #Matriu adjunta
        L_t=np.linalg.inv(Lt)   #invers de l'adjunt

        #Var3.b Calulem \tilde{C}(t)=C(t)
        matrix=np.matmul(L_,pmeanbootB2[n,:,:,t_ref])
        C=np.matmul(matrix,L_t)

        #Var3.c diagonalitzar C i trovar vectors propis \tilde{u}=v i valors propis w (q no utilitzo) per a t_ref
        w, v = np.linalg.eig(C)
        #Var3.d original eigenvectors u per a t_ref
        u=np.matmul(L_t,v)
        #La adjunta de los eigenvectors
        ut=u.T

        #Var3.e trobar la C_alpha,alpha(t)
        for k in range(0,nt):
            matrix=np.matmul(ut,pmeanbootB2[n,:,:,k])
            Cdiag=np.matmul(matrix,u)
            C_diag[n,:,:,k] = Cdiag


#3. Calculem Eb(t) per a los op iguals a la sink i a la source
for op in range(ndimirrep):
    kt=1
    EMpoint=0.
    EMpoint=np.zeros((nboot,(nt-kt)))
    for k in range(0,(nt-kt)):
        for j in range(0,nboot):
            EMpoint[j][k]=np.log(C_diag[j,op,op,k]/C_diag[j,op,op,k+kt])/kt-2*np.log(pmeanbootB1[j,k]/pmeanbootB1[j,k+kt])/kt

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

xboot=list(range(1, nt-kt+1))

plt.rc('text', usetex=True)
plt.rc('font', family='serif', size='12')

fig = plt.figure(figsize=(8,6))

plt.subplots_adjust(left=0.08, bottom=0.08, right=0.98, top=0.95, wspace=0.21, hspace=0.2)
##################            #No cal fer un gràfic cada vegada pero m'ajuda a visualitzar l'ajust -> Cal canviar-ho quan tot vagi be
#PLOT LINEAL
fig1 = fig.add_subplot(1,1,1)
fig1.set_title("Effective mass plot")
fig1.set_ylabel(r'$\mathrm{E} \,\mathrm{(l.u.)}$')
fig1.set_xlabel(r'$t \,\mathrm{(l.u.)}$')
fig1.set_ylim([-0.05,0.1]) #1.10,1.3
fig1.set_xlim([0.5,11.5]) #0,20.5
#plt.xticks([5,10,15,20])
plt.minorticks_on()
fig1.axes.tick_params(which='both',direction='in')
fig1.yaxis.set_ticks_position('both')
fig1.xaxis.set_ticks_position('both')
#Plots dels diferents operadors
for i in range(ndimirrep):
    fig1.errorbar(np.array(xboot)+i*0.1,yboot[i], yerr=eboot[i], ls='None', marker='o', markersize=6, capsize=1, elinewidth=0.7)
#plt.legend()
mn=1.20317
fig1.axhline(0, color='k', linestyle='--')
fig1.axhline(2*np.sqrt(mn**2+(2*np.pi/32)**2)-2*mn, color='k', linestyle='--')
fig1.axhline(2*np.sqrt(mn**2+2*(2*np.pi/32)**2)-2*mn, color='k', linestyle='--')
fig1.axhline(2*np.sqrt(mn**2+3*(2*np.pi/32)**2)-2*mn, color='k', linestyle='--')
fig1.axhline(2*np.sqrt(mn**2+4*(2*np.pi/32)**2)-2*mn, color='k', linestyle='--')

#plt.show()
with PdfPages('B2_I1_A1_singlet_plot.pdf') as pdf:
    pdf.savefig(fig)

plt.close('all')
