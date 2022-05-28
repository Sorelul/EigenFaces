import numpy as np
from numpy import linalg as la
import cv2
import matplotlib
from matplotlib import pyplot as plt
nrPersoane = 40
nrPixeli = 10304  # 122 x 92
nrPozeAntrenare = 8
nrTotalPozeAntrenare = nrPozeAntrenare * nrPersoane
A = np.zeros([nrPixeli, nrTotalPozeAntrenare])
caleaCatreFolder = r'D:\Faculta\Recunoasterea Formelor\Lab\Lab2\att_faces'

global norma
norma = ['1','2','inf','cos']

def creare_satistici_eigen(A,caleaCatreFolder,norma,k):
    A=configurareA()
    correct_responses = 0
    total_tests = 0
    val = preprocesareOptimizata(A,k)
    for folder in range(1,nrPersoane+1):
        for picture in range(9,11):
            imagePath = caleaCatreFolder + fr"\s{folder}/{picture}.pgm"
            persoanaReturnata = cautare(val[0],val[1],val[2],imagePath,norma)
            if persoanaReturnata == folder:
                correct_responses +=1
            total_tests += 1
    print("Total tests: "+ str(total_tests))
    print("Correct responses: "+ str(correct_responses))
    return correct_responses/total_tests

def creare_satistici_eigenRC(A,caleaCatreFolder,norma,k):
    A=configurareA()
    correct_responses = 0
    total_tests = 0
    val = preprocesareReprezentati(A)
    for folder in range(1,nrPersoane+1):
        for picture in range(9,11):
            imagePath = caleaCatreFolder + fr"\s{folder}/{picture}.pgm"
            persoanaReturnata = cautareCuReprezentati(val[0],val[1],val[2],imagePath,norma)
            if persoanaReturnata == folder:
                correct_responses +=1
            total_tests += 1
    print("Total tests: "+ str(total_tests))
    print("Correct responses: "+ str(correct_responses))
    return correct_responses/total_tests

def creare_satistici_nn(A,caleaCatreFolder,norma,k):
    A=configurareA()
    correct_responses = 0
    total_tests = 0
    for folder in range(1,nrPersoane+1):
        for picture in range(9,11):
            imagePath = caleaCatreFolder + fr"\s{folder}/{picture}.pgm"
            persoanaReturnata = nnCuPath(A,imagePath,norma)
            if persoanaReturnata == folder:
                correct_responses +=1
            total_tests += 1
    print("Total tests: "+ str(total_tests))
    print("Correct responses: "+ str(correct_responses))
    return correct_responses/total_tests

def creare_txt_eigen(A,folderPath):
    linii=[]
    for k in range(20,121,20):
        linie = str(k) + ' '
        for norm in range(1,5):
            procent = creare_satistici_eigen(A,folderPath,norm,k)
            linie+=str(procent)+' '
        linii.append(linie)
    with open(r'D:\Faculta\Recunoasterea Formelor\Lab\proiectM2\eigen.txt','w') as f:
        for line in linii:
            f.write(line)
            f.write('\n')

def creare_txt_eigenRC(A,folderPath):
    linii=[]
    for k in range(20,41,20):
        linie = str(k) + ' '
        for norm in range(1,5):
            procent = creare_satistici_eigenRC(A,folderPath,norm,k)
            linie+=str(procent)+' '
        linii.append(linie)
    with open(r'D:\Faculta\Recunoasterea Formelor\Lab\proiectM2\eigenRC.txt','w') as f:
        for line in linii:
            f.write(line)
            f.write('\n')

def creare_txt_nn(A,folderPath):
    linii=[]
    for k in range(20,121,20):
        linie = str(k) + ' '
        for norm in range(1,5):
            procent = creare_satistici_eigen(A,folderPath,norm,k)
            linie+=str(procent)+' '
        linii.append(linie)
    with open(r'D:\Faculta\Recunoasterea Formelor\Lab\proiectM2\nn.txt','w') as f:
        for line in linii:
            f.write(line)
            f.write('\n')


def creare_plots_nn():
    x= []
    contor = 0
    fig, axs = plt.subplots(2,2)
    fig.suptitle('Rata de recunoastere NN')
    with open(r'D:\Faculta\Recunoasterea Formelor\Lab\proiectM2\nn.txt') as f:
        lines = f.readlines()
        for i in range(5):
            x.append([float(line.split()[i]) for line in lines])
            i = i+1
    for i in range (2):
        for j in range (2):
            if contor>1:
                axs[i, j].plot(x[0], x[i + j + 2], marker='o', c='r')
                axs[i, j].set(xlabel='K', ylabel='Acuratete', title='Norma= ' + str(norma[i + j + 1]))
            else:
                axs[i, j].plot(x[0], x[i + j + 1], marker='o', c='r')
                axs[i, j].set(xlabel='K', ylabel='Acuratete', title='Norma= ' + str(norma[i + j]))
                contor+=1
    plt.show()

def creare_plots_eigen():
    x= []
    contor = 0
    fig, axs = plt.subplots(2,2)
    fig.suptitle('Rata de recunoastere EigenFaces')
    with open(r'D:\Faculta\Recunoasterea Formelor\Lab\proiectM2\eigen.txt') as f:
        lines = f.readlines()
        for i in range(5):
            x.append([float(line.split()[i]) for line in lines])
            i = i+1
    for i in range (2):
        for j in range (2):
            if contor>1:
                axs[i, j].plot(x[0], x[i + j + 2], marker='o', c='r')
                axs[i, j].set(xlabel='K', ylabel='Acuratete', title='Norma= ' + str(norma[i + j + 1]))
            else:
                axs[i, j].plot(x[0], x[i + j + 1], marker='o', c='r')
                axs[i, j].set(xlabel='K', ylabel='Acuratete', title='Norma= ' + str(norma[i + j]))
                contor+=1
    plt.show()

def creare_plots_eigenRC():
    x= []
    contor = 0
    fig, axs = plt.subplots(2,2)
    fig.suptitle('Rata de recunoastere EigenFaces Cu Reprezentanti De Clasa')
    with open(r'D:\Faculta\Recunoasterea Formelor\Lab\proiectM2\eigenRC.txt') as f:
        lines = f.readlines()
        for i in range(5):
            x.append([float(line.split()[i]) for line in lines])
            i = i+1
    for i in range (2):
        for j in range (2):
            if contor>1:
                axs[i, j].plot(x[0], x[i + j + 2], marker='o', c='r')
                axs[i, j].set(xlabel='K', ylabel='Acuratete', title='Norma= ' + str(norma[i + j + 1]))
            else:
                axs[i, j].plot(x[0], x[i + j + 1], marker='o', c='r')
                axs[i, j].set(xlabel='K', ylabel='Acuratete', title='Norma= ' + str(norma[i + j]))
                contor+=1
    plt.show()

def cautare(proiectii,HQPB,medie,path,norm):
    A=configurareA()
    pozaCautata=np.array(cv2.imread(path,0))
    pozaCautata = pozaCautata.reshape(10304, )
    pozaCautata = pozaCautata - medie
    pozaCautata = np.dot(pozaCautata, HQPB)
    pozitia = nn(proiectii.T, pozaCautata, norm)  # apel algoritm NN cu norma 1
    return pozitia

def preprocesareOptimizata(A, k):
    A=configurareA()
    medie = np.mean(A, axis=1)
    b = A
    A = (A.T - medie).T
    matCov = np.dot(A.T, A)
    d, v = np.linalg.eig(matCov)
    v = np.dot(A, v)

    indici = np.argsort(d)
    indici = indici[:len(indici) - k - 1:-1]
    print(indici.shape)

    HQPB = v[:, indici]

    proiectii = np.dot(A.T, HQPB)

    A = b

    return proiectii, HQPB, medie

def nn(A, pozaCautata, norm):
    num_rows, num_cols = A.shape
    array_of_all_normed_values = np.array([])

    if norm == 4:
        for every_column in range(num_cols):
            current_value = 1 - np.dot(A[:, every_column], pozaCautata) / (
                    la.norm(A[:, every_column]) * la.norm(pozaCautata))
            array_of_all_normed_values = np.append(array_of_all_normed_values, current_value)
    else:
        if norm == 3:
            norm = np.inf
        for every_column in range(num_cols):
            current_value = la.norm(A[:, every_column] - pozaCautata, norm)
            array_of_all_normed_values = np.append(array_of_all_normed_values, current_value)

    min_position = np.argmin(array_of_all_normed_values)
    return (min_position // 8) + 1

def nnCuPath(A, pozaCautata, norm):
    num_rows, num_cols = A.shape
    array_of_all_normed_values = np.array([])
    img = cv2.imread(pozaCautata, 0)
    img = np.array(img)
    img = img.reshape(-1, )
    if norm == 4:
        for every_column in range(num_cols):
            current_value = 1 - np.dot(A[:, every_column], img) / (
                    la.norm(A[:, every_column]) * la.norm(img))
            array_of_all_normed_values = np.append(array_of_all_normed_values, current_value)
    else:
        if norm == 3:
            norm = np.inf
        for every_column in range(num_cols):
            current_value = la.norm(A[:, every_column] - img, norm)
            array_of_all_normed_values = np.append(array_of_all_normed_values, current_value)

    min_position = np.argmin(array_of_all_normed_values)
    return (min_position // 8) + 1

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

def cautareCuReprezentati(media, HQPB, proiectii,path,norm):
    A=configurareA()
    pozaCautata = np.array(cv2.imread(path, 0))
    pozaCautata = pozaCautata.reshape(10304, )
    pozaCautata = pozaCautata - media
    pozaCautata = np.dot(pozaCautata, HQPB)
    pozitia = nn(proiectii.T, pozaCautata, norm)  # apel algoritm NN cu norma 1
    return pozitia

def preprocesareReprezentati(a):
    rc = np.zeros([nrPixeli, 40])
    for i in range(40):
        b=a[:][(i*nrPozeAntrenare):(i+1)*nrPozeAntrenare]
        rc[:, i] = np.mean(a[:, (i * 8):((i + 1) * 8)], axis=1)
    media = np.mean(rc, axis=1)
    rc = (rc.T - media).T
    c = np.dot(rc.T, rc)
    d, v = np.linalg.eig(c)
    v = np.dot(rc, v)
    indici = np.argsort(d)
    HQPB = v[:, indici]
    proiectii = np.dot(rc.T, HQPB)
    return media, HQPB, proiectii


# creare_txt_eigenRC(configurareA(),caleaCatreFolder)