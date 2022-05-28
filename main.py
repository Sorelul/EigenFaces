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
    global A
    for i in range(1, nrPersoane + 1):
        caleaCatrePersoana = caleaCatreFolder + '\s' + str(i) + '\\'
        for j in range(1, nrPozeAntrenare + 1):
            calePozaAntrenare = caleaCatrePersoana + str(j) + '.pgm'
            # citim poza ca matrice 112 x 92:
            pozaAntrenare = np.array(cv2.imread(calePozaAntrenare, 0))
            # vectorizam poza:
            pozaVect = pozaAntrenare.reshape(-1, )
            A[:, nrPozeAntrenare * (i - 1) + j - 1] = pozaVect



def cautare():
    configurareA()
    proiectii,HQPB,medie = preprocesareOptimizata(A, k)
    pozaCautata = np.array(cv2.imread(r'D:\Faculta\Recunoasterea Formelor\Lab\Lab2\att_faces\s39\10.pgm', 0))
    plt.imshow(pozaCautata, cmap='gray', vmin=0, vmax=255)
    plt.show()
    pozaCautata = pozaCautata.reshape(10304, )
    pozaCautata = pozaCautata - medie
    pozaCautata = np.dot(pozaCautata, HQPB)
    pozitia = nn(proiectii.T, pozaCautata, '1')  # apel algoritm NN cu norma 1
    plt.imshow(A[:, pozitia].reshape(112, 92), cmap='gray', vmin=0, vmax=255)
    plt.show()
    print(pozitia)
    return pozitia

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


def preprocesareOptimizata(A, k):
    configurareA()
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


# k = input("Introduceti k:")
# while k not in ('20', '40', '60', '80', '100', '120'):
#     k = input("Alegeti una din valorile: 20, 40, 60, 80, 100, 120")
# else:
#     k = int(k)
#     # proiectii, HQPB, medie = preprocesareOptimizata(A, k)
#     #cautare_lanczos()
#     cautare()
#     #cautareCuReprezentati()
