import sys
from PyQt5 import QtWidgets
from PyQt5 import uic
from PyQt5.QtGui import *
from PyQt5.QtWidgets import (QLabel, QRadioButton, QPushButton, QVBoxLayout, QApplication, QWidget, QFileDialog)
import numpy as np
from numpy import linalg as la
import cv2
import matplotlib
from matplotlib import pyplot as plt
from main import cautare as caut

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


def paginaPrinc():

    if formular.orlButton.isChecked():
        print("Ai selectionat baza de date ORL ")

    elif formular.essexButton.isChecked():
        print("Ai selectionat baza de date essex ")

    elif formular.ctovfButton.isChecked():
        print("Ai selectionat baza de date CTOVF ")

    #Alegem numarul de poze de antrenare
    global nrPozeAntrenare
    if formular.antrenare1.isChecked():
        nrPozeAntrenare = 6
        print("60 cu 40")
    elif formular.antrenare2.isChecked():
        nrPozeAntrenare = 8
        print("80 cu 20")
    elif formular.antrenare3.isChecked():
        nrPozeAntrenare = 9
        print("90 cu 10")

    global k
    k = formular.kValComboBox.currentText()
    print(k)

    #Alegem algoritmul
    global algoritmFolosit
    if formular.nnButton.isChecked():
        algoritmFolosit='NN'
        print("Am selectat nn")
    elif formular.knnButton.isChecked():
        algoritmFolosit = 'KNN'
        print("Am selectat knn")
    elif formular.eigenFacesOptimizatButton.isChecked():
        algoritmFolosit = 'EigenFaces'
        print("Am selectat eigen")
    elif formular.eigenFacesButton.isChecked():
        algoritmFolosit = 'EigenFacesRC'
        print("Am selectat eigenfaces")
    elif formular.lanczosButton.isChecked():
        algoritmFolosit = 'Lanczos'
        print("Am selectat lanczos")

    if formular.norma1Button.isChecked():
        normaFolosita = '1'
        print(normaFolosita)
    elif formular.norma2Button.isChecked():
        normaFolosita = '2'
        print(normaFolosita)
    elif formular.normaInfinitButton.isChecked():
        normaFolosita = 'inf'
        print(normaFolosita)
    elif formular.normaCosButton.isChecked():
        normaFolosita = 'cos'
        print(normaFolosita)

def umplereValCombo():
    formular.kValComboBox.addItem("20")
    formular.kValComboBox.addItem("40")
    formular.kValComboBox.addItem("60")
    formular.kValComboBox.addItem("80")
    formular.kValComboBox.addItem("100")
    formular.kValComboBox.addItem("120")

def cautaPoza():
    filename = QFileDialog.getOpenFileName()
    global path
    path = filename[0]
    pixmap = QPixmap(path)
    formular.labelImagine.setPixmap(pixmap)


def norma_dif(x, y, norma):
    cases = {
        '1': la.norm(x - y, 1),
        '2': la.norm(x - y),
        'inf': la.norm(x - y, np.inf),
        'cos': 1 - np.dot(x, y) / (la.norm(x) * la.norm(y))
    }
    return cases.get(norma)

def nn(A, pozaCautata, norma):
    z = np.zeros(len(A[0]))
    for i in range(len(z)):
        z[i] = norma_dif(A[:, i], pozaCautata, norma)
    pozitia = np.argmin(z)
    return pozitia

def configurareA():

    for i in range(1, nrPersoane + 1):
        caleaCatrePersoana = caleaCatreFolder + '\s' + str(i) + '\\'
        for j in range(1, nrPozeAntrenare + 1):
            calePozaAntrenare = caleaCatrePersoana + str(j) + '.pgm'
            # citim poza ca matrice 112 x 92:
            pozaAntrenare = np.array(cv2.imread(calePozaAntrenare, 0))
            # vectorizam poza:
            pozaVect = pozaAntrenare.reshape(-1, )
            A[:, nrPozeAntrenare * (i - 1) + j - 1] = pozaVect


def preprocesareOptimizata(A,k):

    medie = np.mean(A, axis=1)
    b = A
    A = (A.T - medie).T

    matCov = np.dot(A.T, A)
    d, v = np.la.eig(matCov)
    # print(path)
    v = np.dot(A, v)

    indici = np.argsort(d)
    indici = indici[:len(indici) - k - 1:-1]
    print(indici.shape)

    HQPB = v[:, indici]

    proiectii = np.dot(A.T, HQPB)

    A = b

    return proiectii, HQPB, medie

def preprocesarelanczos(a, k):
    beta = 0
    q = np.zeros([nrPixeli, k + 2])
    q[:, 0] = np.zeros(nrPixeli)
    q[:, 1] = np.ones(nrPixeli)
    q[:, 1] = q[:, 1] / la.norm(q[:, 1])  # Aproximari ale fantomelor
    for i in range(1, k + 1):
        w = np.dot(a, (np.dot(a.T, q[:, i]))) - beta * q[:, i - 1]
        alpha = np.dot(w, q[:, i])
        w = w - alpha * q[:, i]
        beta = la.norm(w, 2)
        q[:, i + 1] = w / beta
    hqpb = q[:, 2:]
    proiectii = np.dot(a.T, hqpb)
    return hqpb, proiectii

def cautare_lanczos():
    configurareA()
    HQPB, proiectii = preprocesarelanczos(A,40)
    # testare poza 9 a persoanei 40:
    pozaCautata = np.array(cv2.imread(
        r'D:\Faculta\Recunoasterea Formelor\Lab\Lab2\att_faces\s40\9.pgm', 0))
    plt.imshow(pozaCautata, cmap='gray', vmin=0, vmax=255)
    plt.show()
    pozaCautata = pozaCautata.reshape(10304, )
    pozaCautata = np.dot(pozaCautata, HQPB)
    pozitia = nn(proiectii.T, pozaCautata, '1')  # apel algoritm NN cu norma 1
    plt.imshow(A[:, pozitia].reshape(112, 92), cmap='gray', vmin=0, vmax=255)
    plt.show()
    plt.show()
    print(pozitia)

def cautare():
    configurareA()
    proiectii,HQPB,medie = preprocesareOptimizata(A,k)
    pozaCautata = np.array(cv2.imread(path,0))
    pozaCautata = pozaCautata.reshape(10304, )
    pozaCautata = pozaCautata - medie
    pozaCautata = np.dot(pozaCautata, HQPB)
    pozitia = nn(proiectii.T, pozaCautata, norma)  # apel algoritm NN cu norma 1
    formular.labelImagineGasita.setPixmap(A[:, pozitia].reshape(112, 92))
    print(pozitia)



app=QtWidgets.QApplication([])
formular=uic.loadUi("incercare_gui2.ui")



umplereValCombo()
formular.pushButton.clicked.connect(paginaPrinc)
formular.selecteazaButton.clicked.connect(cautaPoza)




formular.show()
app.exec()