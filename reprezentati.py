import numpy as np
from numpy import linalg as la
import cv2  # pentru citirea pozelor
import matplotlib
from matplotlib import pyplot as plt  # pentru grafice

nrPersoane = 40
nrPixeli = 10304  # 122 x 92
nrPozeAntrenare = 8
nrTotalPozeAntrenare = nrPozeAntrenare * nrPersoane
A = np.zeros([nrPixeli, nrTotalPozeAntrenare])
caleaCatreFolder = r'D:\Faculta\Recunoasterea Formelor\Lab\Lab2\att_faces'

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

def cautareCuReprezentati():
    configurareA()
    media, HQPB, proiectii = preprocesareReprezentati(A)
    # testare poza 9 a persoanei 40:
    pozaCautata = np.array(cv2.imread(
        r'D:\Faculta\Recunoasterea Formelor\Lab\Lab2\att_faces\s40\10.pgm', 0))
    plt.imshow(pozaCautata, cmap='gray', vmin=0, vmax=255)
    plt.show()
    pozaCautata = pozaCautata.reshape(10304, )
    pozaCautata = pozaCautata - media
    pozaCautata = np.dot(pozaCautata, HQPB)
    pozitia = nn(proiectii.T, pozaCautata, '1')  # apel algoritm NN cu norma 1
    plt.imshow(A[:, (pozitia+1)*8-1].reshape(112, 92), cmap='gray', vmin=0, vmax=255)
    plt.show()
    plt.show()
    print(pozitia)


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

# k = input("Introduceti k:")
# while k not in ('20', '40', '60', '80', '100', '120'):
#     k = input("Alegeti una din valorile: 20, 40, 60, 80, 100, 120")
# else:
#     k = int(k)
#     cautareCuReprezentati()