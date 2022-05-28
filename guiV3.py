import sys
from PyQt5 import QtWidgets
from PyQt5 import uic
from PyQt5.QtGui import *
from PyQt5.QtWidgets import (QLabel, QRadioButton, QPushButton, QVBoxLayout, QApplication, QWidget, QFileDialog)
import numpy as np
from numpy import linalg as la
import cv2
import statistics as st
import statistici
import matplotlib
from matplotlib import pyplot as plt

nrPersoane = 40
nrPixeli = 10304
nrPozeAntrenare = 6
algoritmFolosit = 'NN'
normaFolosita = '2'
nrTotalPozeAntrenare = nrPozeAntrenare * nrPersoane
A = np.zeros([nrPixeli, nrTotalPozeAntrenare])
caleaCatreFolder = r'D:\Faculta\Recunoasterea Formelor\Lab\Lab2\att_faces'
calePozaCautata = caleaCatreFolder+'\s10\8.pgm'
norme=['1','2','inf','cos']
poza_cautata = np.array(cv2.imread(calePozaCautata,0))
pozaCautataVector = poza_cautata.reshape(nrPixeli,)
B=A


def HomePage():

    if guiul.orlButton.isChecked():
        print("Ai selectionat baza de date ORL ")

    elif guiul.essexButton.isChecked():
        print("Ai selectionat baza de date essex ")

    elif guiul.ctovfButton.isChecked():
        print("Ai selectionat baza de date CTOVF ")

    #Alegem numarul de poze de antrenare
    global nrPozeAntrenare
    if guiul.antrenare1.isChecked():
        nrPozeAntrenare = 6
        print("60 cu 40")
    elif guiul.antrenare2.isChecked():
        nrPozeAntrenare = 8
        print("80 cu 20")
    elif guiul.antrenare3.isChecked():
        nrPozeAntrenare = 9
        print("90 cu 10")

    #Alegem algoritmul
    global algoritmFolosit
    if guiul.nnButton.isChecked():
        algoritmFolosit='NN'
        print("Am selectat nn")
    elif guiul.knnButton.isChecked():
        algoritmFolosit = 'KNN'
        print("Am selectat knn")
    elif guiul.eigenFacesOptimizatButton.isChecked():
        algoritmFolosit = 'EigenFaces'
        print("Am selectat eigen")
    elif guiul.eigenFacesButton.isChecked():
        algoritmFolosit = 'EigenFacesRC'
        print("Am selectat eigenfaces")
    elif guiul.lanczosButton.isChecked():
        algoritmFolosit = 'Lanczos'
        print("Am selectat lanczos")

    if guiul.norma1Button.isChecked():
        normaFolosita = '1'
        print(normaFolosita)
    elif guiul.norma2Button.isChecked():
        normaFolosita = '2'
        print(normaFolosita)
    elif guiul.normaInfinitButton.isChecked():
        normaFolosita = 'inf'
        print(normaFolosita)
    elif guiul.normaCosButton.isChecked():
        normaFolosita = 'cos'
        print(normaFolosita)

def cautaPrimaPoza():
    filename = QFileDialog.getOpenFileName()
    global path
    path = filename[0]
    pixmap = QPixmap(path)
    pixmap.scaledToWidth(300)
    guiul.labelImagine.setPixmap(pixmap)

def configurareA():

    global A,B,nrPersoane,nrTotalPozeAntrenare
    nrTotalPozeAntrenare=nrPersoane*nrPozeAntrenare
    A = np.zeros([nrPixeli,nrTotalPozeAntrenare])
    k=0
    for i in range(1,nrPersoane+1):
        caleaCatrePersoana = caleaCatreFolder + '\s' + str(i) + '\\'
        for j in range(1,nrPozeAntrenare+1):
            calePozaAntrenare = caleaCatrePersoana + str(j) + '.pgm'
            pozaAntrenare = np.array(cv2.imread(calePozaAntrenare,0))
            pozaVect=pozaAntrenare.reshape(nrPixeli,)
            A[:,k]=pozaVect
            k+=1
    B=A
    return A

def cautaPoza():
    knnK = int(guiul.knnKcomboBox.currentText())
    eigenK = int(guiul.eigenKcomboBox.currentText())
    lanczosK = int(guiul.lanczosKcomboBox.currentText())
    A=configurareA()
    print('cautam poza: ',calePozaCautata)
    print('algoritmul folosit este: ',algoritmFolosit)
    print('norma folosita este ',normaFolosita)
    print('Numarul de persoane este: ',nrPersoane)
    print('Numarul de poze este: ',nrPozeAntrenare)
    print(calePozaCautata)
    poza_cautata = np.array(cv2.imread(path,0))
    pozaCautataVector = poza_cautata.reshape(nrPixeli,)
    if algoritmFolosit == 'NN':
        pozitia = NN(A,pozaCautataVector,normaFolosita)
    elif algoritmFolosit == 'KNN':
        pozitia = KNN(A, pozaCautataVector, normaFolosita,knnK)
    elif algoritmFolosit == 'EigenFaces':
        pozitia = EIGEN(A, pozaCautataVector, normaFolosita,eigenK)
    elif algoritmFolosit == 'EigenFacesRC':
        pozitia = EIGENRC(A, pozaCautataVector, normaFolosita,eigenK)
    elif algoritmFolosit == 'Lanczos':
        pozitia = Lanczos(A, pozaCautataVector, normaFolosita,lanczosK)
    else:
        print('nici un algoritm selectat')
    pozitia =(pozitia//nrPozeAntrenare)+1
    pozitiaPozei = pozitia%nrPozeAntrenare
    if pozitiaPozei == 0:
        pozitiaPozei=1
    print('persoana: ',pozitia)
    print('poza: ',pozitiaPozei)
    imagineDePrintat = caleaCatreFolder+'\s'+str(pozitia)+'\\'+str(pozitiaPozei)+'.pgm'
    guiul.labelImagineGasita.setPixmap(QPixmap(imagineDePrintat))

def NN(A,poza_cautata,norma):
    norma = str(norma)
    poza_cautata = poza_cautata.reshape(-1,)
    z =np.zeros(([1,len(A[0])]), dtype=float)
    for i in range(0,len(A[0])):
        if norma == '1':
            diferenta = poza_cautata-A[:,i]
            z[0,i]=la.norm(diferenta,1)
        elif norma == '2':
            diferenta = poza_cautata - A[:, i]
            z[0, i] = la.norm(diferenta, 2)
        elif norma == 'inf':
            diferenta = poza_cautata - A[:, i]
            z[0, i] = la.norm(diferenta, np.inf)
        elif norma == 'cos':
            numarator = np.dot(poza_cautata,A[:,i])
            numitor = la.norm(poza_cautata)*la.norm(A[:,i])
            z[0, i] = (1-numarator)/numitor
        else:
            exit()
    return np.argmin(z)

def KNN(A,poza_cautata,norma,k):
    poza_cautata = poza_cautata.reshape(-1, )
    z = np.zeros(([1, len(A[0])]), dtype=float)
    for i in range(0, len(A[0])):
        if norma == '1':
            diferenta = poza_cautata-A[:,i]
            z[0,i]=la.norm(diferenta,1)
        elif norma == '2':
            diferenta = poza_cautata - A[:, i]
            z[0, i] = la.norm(diferenta, 2)
        elif norma == 'inf':
            diferenta = poza_cautata - A[:, i]
            z[0, i] = la.norm(diferenta, np.inf)
        elif norma == 'cos':
            numarator = np.dot(poza_cautata,A[:,i])
            numitor = la.norm(poza_cautata)*la.norm(A[:,i])
            z[0, i] = (1-numarator)/numitor
        else:
            exit()
    pozitii = np.argsort(z)
    pozitii=pozitii[:k]
    vecini=np.zeros(len(pozitii),)
    for i in range(0,len(pozitii)):
        if pozitii[i]%nrPozeAntrenare ==0:
            vecini[i] = pozitii[i]/nrPozeAntrenare
        else:
            vecini[i]=pozitii[i]//nrPozeAntrenare+1
    vecin=int(st.mode(vecini))
    pozitie=nrPozeAntrenare*(vecin-1)
    return pozitie

def EIGEN(A,pozaCautataVector,norma,k):

    media = np.mean(A, axis=1)

    A = (A.T - media).T
    # C=np.dot(A,A.T)
    L = np.dot(A.T, A)
    d, v = np.linalg.eig(L)
    v = np.dot(A, v)

    vsorted = np.argsort(d)
    vsorted = vsorted[-1:-k - 1:-1]
    HQPB = np.zeros([nrPixeli, k])
    for n in range(0, k):
        # print(vsorted[n])
        HQPB[:, n] = v[:, vsorted[n]]
    proiectii = np.dot(A.T, HQPB)

    pozacautata = pozaCautataVector - media
    proiectie_cautata = np.dot(pozacautata, HQPB)
    return NNEIG(proiectii.T,proiectie_cautata,norma)

def NNEIG(A,poza_cautata,norma):
    z=np.zeros(([1,len(A[0])]),dtype=float)
    for i in range(0,len(A[0])):
        if norma == '1':
            diferenta = poza_cautata - A[:, i]
            z[0, i] = la.norm(diferenta, 1)
        elif norma == '2':
            diferenta = poza_cautata - A[:, i]
            z[0, i] = la.norm(diferenta, 2)
        elif norma == 'inf':
            diferenta = poza_cautata - A[:, i]
            z[0, i] = la.norm(diferenta, np.inf)
        elif norma == 'cos':
            numarator = np.dot(poza_cautata, A[:, i])
            numitor = la.norm(poza_cautata) * la.norm(A[:, i])
            z[0, i] = (1 - numarator) / numitor
        else:
            exit()
    return np.argmin(z)

def EIGENRC(A,pozaCautataVector,norma,k):
    RC = np.zeros([nrPixeli, nrPersoane])
    for t in range(0, nrPersoane):
        start = t * nrPozeAntrenare
        RC[:, t] = np.mean(A[:, start:start + nrPozeAntrenare], axis=1)
    A = RC
    media = np.mean(A, axis=1)
    A = (A.T - media).T
    L = np.dot(A.T, A)
    d, v = np.linalg.eig(L)
    v = np.dot(A, v)
    vsorted = np.argsort(d)
    vsorted = vsorted[-1:-k - 1:-1]
    HQPB = np.zeros([nrPixeli, k])
    for n in range(0, k):
        HQPB[:, n] = v[:, vsorted[n]]
    proiectii = np.dot(A.T, HQPB)
    pozacautata = pozaCautataVector - media
    proiectie_cautata = np.dot(pozacautata, HQPB)
    nn=NNRC(proiectii.T,proiectie_cautata,norma)
    return(nn*nrPozeAntrenare)

def NNRC(A,poza_cautata,norma):
    z = np.zeros(([1, len(A[0])]), dtype=float)
    for i in range(0, len(A[0])):
        if norma == '1':
            diferenta = poza_cautata - A[:, i]
            z[0, i] = la.norm(diferenta, 1)
        elif norma == '2':
            diferenta = poza_cautata - A[:, i]
            z[0, i] = la.norm(diferenta, 2)
        elif norma == 'inf':
            diferenta = poza_cautata - A[:, i]
            z[0, i] = la.norm(diferenta, np.inf)
        elif norma == 'cos':
            numarator = np.dot(poza_cautata, A[:, i])
            numitor = la.norm(poza_cautata) * la.norm(A[:, i])
            z[0, i] = (1 - numarator) / numitor
        else:
            exit()
    return np.argmin(z)

def Lanczos_puiut(A, k=20):
    q = np.zeros([nrPixeli, k+ 2])
    q[:, 0] = np.zeros(nrPixeli)
    q[:, 1] = np.ones(nrPixeli)
    q[:, 1] = q[:,1]/la.norm(q[:,1])
    beta = 0
    for i in range(1, k + 1):
        w = np.dot(A, np.dot(A.T, q[:, i])) - np.dot(beta, q[:, i - 1])
        alpha = np.dot(w, q[:, i])
        w = w - np.dot(alpha, q[:, i])
        beta = la.norm(w, 2)
        q[:, i] = w / beta
    return q[:, 2:k + 1]

def Lanczos(A,pozaCautataVector,norma,k):
    HQPB = Lanczos_puiut(A, k)
    proiectii = np.dot(A.T, HQPB)
    A = proiectii.T
    poza_cautata = np.dot(pozaCautataVector, HQPB)
    z = np.zeros(([1, len(A[0])]), dtype=float)
    for i in range(0, len(A[0])):
        if norma == '1':
            diferenta = poza_cautata - A[:, i]
            z[0, i] = la.norm(diferenta, 1)
        elif norma == '2':
            diferenta = poza_cautata - A[:, i]
            z[0, i] = la.norm(diferenta, 2)
        elif norma == 'inf':
            diferenta = poza_cautata - A[:, i]
            z[0, i] = la.norm(diferenta, np.inf)
        elif norma == 'cos':
            numarator = np.dot(poza_cautata, A[:, i])
            numitor = la.norm(poza_cautata) * la.norm(A[:, i])
            z[0, i] = (1 - numarator) / numitor
        else:
            exit()
    return np.argmin(z)


app=QtWidgets.QApplication([])
guiul=uic.loadUi("incercare_gui2.ui")
guiul.pushButton.clicked.connect(HomePage)
guiul.selecteazaButton.clicked.connect(cautaPrimaPoza)
guiul.cautareButton.clicked.connect(cautaPoza)
guiul.statisticiButton.clicked.connect(statistici.creare_plots_nn)
guiul.eigenfaces.clicked.connect(statistici.creare_plots_eigen)
guiul.eigenfacesRC.clicked.connect(statistici.creare_plots_eigenRC)
guiul.show()
app.exec()