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
#n_op=10  #10
n_oplist = [3,4,5,6]
n_op = len(n_oplist)
n_smear = 1
ndim = n_op*n_smear
yboot=[]
eboot=[]
E_b=[]
#op=3,11
#DELS FITXER .H5
fh5 = h5py.File('/Users/sandra/Documents/GitHub/TFM/qblocks_matrix_irreps_cl3_32_48_b6p1_m0p2450_frontera.h5', 'r')
#fh5 = h5py.File('/Volumes/GoogleDrive-100962958533814099960/.shortcut-targets-by-id/1mV9ITzWE4JpeOWAX6GdzvZ5l3if1bz8j/Sandra Tomás/Codis/data/qblocks_matrix_irreps_cl3_32_48_b6p1_m0p2450_frontera.h5', 'r')
blck = np.zeros((nsc,ndim,ndim,nt))
for i,opsrc in enumerate(n_oplist):
    for j,opsnk in enumerate(n_oplist):
        for ii in range(n_smear):
            for jj in range(n_smear):
                iii = i+ii*n_op
                jjj = j+jj*n_op
                blck[:,iii,jjj,:] = 0.5*(np.real(np.array(fh5['B2_I1_A1_f'][0:nsc,opsrc,ii,opsnk,jj,0:nt])+np.real(np.array(fh5['B2_I1_A1_b'][0:nsc,opsrc,ii,opsnk,jj,0:nt]))))

blck2=np.zeros((nsc,ndim,ndim,nt))
for op_src in range(ndim):
    for op_snk in range(ndim):
        #Var1 pre-processing antes de diagonalitzar C_ab(t)
        for k in range(nt):
            blck2[:,op_src,op_snk,k]=0.5*(blck[:,op_src,op_snk,k]+blck[:,op_snk,op_src,k])/np.sqrt(blck[:,op_src,op_src,0]*blck[:,op_snk,op_snk,0])

#Var1.2 Les matrius C_b(t)=pmeanboot correlator samples bootstrap samples NxN=[opsrc][opsnk]
#Millor passar array de op especifics
pmeanboot=np.zeros((nboot,ndim,ndim,nt))
x=np.random.uniform(size=(nsc,nboot))  #Matriu de num aleatoris entre 0 i 1
for j in range(nboot):
    boot=np.zeros((ndim,ndim,nt))
    for i in range(nsc):
        boot=boot+blck2[int(x[i][j]*nsc),:,:,:]
    pmeanboot[j,:,:,:]=boot*nsc_   #Ara hem generat les Nb bootstrap samples Cb(t)

#pmeanboot=C(t)
#C=\tilde{C}(t)
#C_diag=matriu final C_alpha,alpha que utilitzem per trobar la energia
#Var3.a find the cholesky decomposition of pmeanboot2(t0)
t_0=4-1 #t0=4
#Tot per t_ref=2*t_0=8
t_ref=2*(t_0+1)-1   #El -1 es per la llista q comensa en 0
C_diag=np.zeros((nboot,ndim,ndim,nt))
for n in range(nboot):
    L=np.linalg.cholesky(pmeanboot[n,:,:,t_0])
    L_=np.linalg.inv(L) #Matriu inversa
    Lt=L.T   #Matriu adjunta
    L_t=np.linalg.inv(Lt)   #invers de l'adjunt

    #Var3.b Calulem \tilde{C}(t)=C(t)
    matrix=np.matmul(L_,pmeanboot[n,:,:,t_ref])
    C=np.matmul(matrix,L_t)

    #Var3.c diagonalitzar C i trovar vectors propis \tilde{u}=v i valors propis w (q no utilitzo) per a t_ref
    w, v = np.linalg.eig(C)
    #Var3.d original eigenvectors u per a t_ref
    u=np.matmul(L_t,v)
    #La adjunta de los eigenvectors
    ut=u.T

    #Var3.e trobar la C_alpha,alpha(t)
    for k in range(0,nt):
        matrix=np.matmul(ut,pmeanboot[n,:,:,k])
        Cdiag=np.matmul(matrix,u)
        C_diag[n,:,:,k] = Cdiag

    #print(C_diag[n,:,:,k])
kt=1
#3. Calculem Eb(t) per a los op iguals a la sink i a la source

for op in range(ndim):
    kt=1
    EMpoint=0.
    EMpoint=np.zeros((nboot,(nt-kt)))
    for k in range(0,(nt-kt)):
        for j in range(0,nboot):
            EMpoint[j][k]=np.log(C_diag[j,op,op,k]/C_diag[j,op,op,k+kt])/kt

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

    e_b=np.zeros(((nt-kt),nboot))
    for k in range(0,(nt-kt)):
        for j in range(0,nboot):
            e_b[k][j]=EMpoint[j][k]

    #Dades
    yboot.append(mean)
    eboot.append(sigm)
    E_b.append(e_b)

xboot=list(range(1, nt))
