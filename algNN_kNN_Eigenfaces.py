import numpy as np
from numpy import linalg as la
import cv2 # pentru citirea pozelor
import matplotlib
from matplotlib import pyplot as plt # pentru grafice
import statistics as st

nrPersoane=40
nrPixeli=10304
nrPozeAntrenare=8
nrTotalPozeAntrenare=nrPozeAntrenare*nrPersoane
A = np.zeros([nrPixeli, nrTotalPozeAntrenare])


def norma_dif(x, y, norma):
    cases = {
        '1': la.norm(x - y, 1),
        '2': la.norm(x - y),
        'inf': la.norm(x - y, np.inf),
        'cos': 1 - np.dot(x, y) / (la.norm(x) * la.norm(y))
    }
    return cases.get(norma)


def nn(A,pozaCautata,norma):
	z=np.zeros(len(A[0]))
	for i in range(len(z)):
		z[i]=norma_dif(A[:,i],pozaCautata,norma)
	pozitia=np.argmin(z)
	return pozitia

def knn(A,pozaCautata,norma):
	global nrPozeAntrenare
	global k
	nrPozeAntrenare=int(nrPozeAntrenare)
	k = int(k)
	z=np.zeros(len(A[0]))
	for i in range(len(z)):
		z[i]=norma_dif(A[:,i],pozaCautata,norma)
	pozitii=np.argsort(z)
	pozitii=pozitii[:k]
	vecini=pozitii//nrPozeAntrenare
	vecin=int(st.mode(vecini))
	return vecin*nrPozeAntrenare

def preprocEigenfaces(A,pozaCautata,norma):
	global k
	global media
	global HQPB
	global proiectii
	print("Preprocesare Eigenfaces: ")
	k = int(k)
	media = np.mean(A,axis=1)
	A=(A.T-media).T
	L=np.dot(A.T,A)
	d,v=la.eig(L)
	v=np.dot(A,v)
	indx=np.argsort(d)
	indx=indx[:-k-1:-1]
	HQPB=v[:,indx]
	proiectii=np.dot(A.T,HQPB)

def interEigenfaces(A,pozaCautata,norma):
	global media
	global HQPB
	global proiectii
	print("Interogare Eigenfaces: ")
	pozaCautata=pozaCautata-media
	prCautata=np.dot(pozaCautata.T,HQPB)
	pozitia=nn(proiectii.T,prCautata,norma)
	return pozitia

def configurareA():
	caleBD=r'D:\ACS\proiect\frweb\att_faces'
	for i in range(1,nrPersoane+1):
		caleFolderPers=caleBD+'\s'+str(i)+'\\'
		for j in range(1,nrPozeAntrenare+1):
			calePozaAntrenare=caleFolderPers+str(j)+'.pgm'
			# citim poza ca matrice 112 x 92:
			pozaAntrenare=np.array(cv2.imread(calePozaAntrenare,0))
			# vectorizam poza:
			pozaVect=pozaAntrenare.reshape(10304,) 
			A[:,nrPozeAntrenare*(i-1)+j-1] = pozaVect

def cautare():
	# testare poza 9 a persoanei 40:
	pozaCautata=np.array(cv2.imread(r'D:\ACS\proiect\frweb\att_faces\s40\9.pgm',0))
	plt.imshow(pozaCautata, cmap='gray', vmin=0, vmax=255)
	plt.show()
	pozaCautata=pozaCautata.reshape(10304,)
	pozitia=nn(A,pozaCautata,'1') # apel algoritm NN cu norma 1
	plt.imshow(A[:,pozitia].reshape(112,92), cmap='gray', vmin=0, vmax=255)
	plt.show()