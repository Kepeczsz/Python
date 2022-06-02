from random import *
import numpy as np
import pandas as pd
#zad1
from numpy.lib.stride_tricks import as_strided

x = np.random.randint(0,100,(10,5))


print("diag: ",np.diag(x))
print("trace: ",np.trace(x))

#zad2
arr2 = np.array(np.random.normal(size = (5,5)).round(2))
arr3 = np.array(np.random.normal(size = (5,5)).round(2))

arr4 = np.dot(arr2,arr3)
print("Zadanie2:", arr4)
#zad3
ai = randrange(0, 101, 5)
aj = randrange(0, 101, 5)

arr5 = np.random.randint(0, 101, size=(ai, aj))
arr6 = np.random.randint(0, 101, size=(ai, aj))
z = np.reshape(arr5, (-1, 5))
z1 = np.reshape(arr6, (-1, 5))

print("zad3\n",)
print(np.add(z,z1))


#zad4

l1 = np.zeros([5,4])
l2 = np.zeros([4,5])
arr1 = l2.transpose()
c2 = np.zeros([5,4])
#zad4
arr7 = np.random.randint(0,100,(4,5))
arr8 = np.random.randint(0,100,(5,4))
c1 = arr8.transpose()
c2 = np.add(arr7,c1)
print("zad4\n", c2)

#zad5
print("zad5\n")
arr9 = np.random.randint(0,100,(10, 10))
arr10 = np.random.randint(0,100,(10, 10))

l3 = np.multiply(arr10[:,3],arr9[:,2])
print("wynik: ", l3)


#zad6
print("zad6")
arr2 = np.array(np.random.normal(size = (4,4)).round(3))
arr3 = np.array(np.random.uniform(size=(3,3)).round(3))
print("wartosc srednia 1 macierzy: ", np.mean(arr2))
print("odchylenie standardowe 1 macierzy:", np.std(arr2))
print("Wariancja 1 macierzy", np.var(arr2))
print("suma macierzy 1", np.sum(arr2))
print("Wartosc maksymalna 1 macierzy", np.max(arr2))
print("Wartosc minimalna 1 macierzy", np.min(arr2))

print("wartosc srednia 2 macierzy: ", np.mean(arr3))
print("odchylenie standardowe 2 macierzy:", np.std(arr3))
print("Wariancja 2 macierzy", np.var(arr3))
print("Suma 2 macierzy ", np.sum(2))
print("Wartosc maksymalna 2 macierzy", np.max(arr3))
print("Wartosc minimalna 2 macierzy", np.min(arr3))

#zad7

print("zad7")

arr4 = np.array(np.random.random(size = (4,4)))
arr5 = np.array(np.random.random(size = (4,4)))
arr6 = np.zeros([4, 4])
arr7 = np.zeros([4, 4])
arr6 = arr4*arr5
arr7 = np.dot(arr4, arr5)

print(arr6)
print(arr7)
print("roznica jest taka, ze funkcja dot wykonuje poprawne mnozenie macierzowe, a operator '*' mnozy element razy element")

#zad8
print("Zad8")
arr11 = np.random.randint(0,100,(5, 5))
print(as_strided(x, shape=(15,), strides=(4,)))
#zad9
arr12 = np.random.randint(0,100,(3, 3))
arr13 = np.random.randint(0,100,(3, 3))
a = np.vstack((arr12,arr13))
print("zad9\n vstack\n", a )
arr14 = np.stack((arr12, arr13))
print("stack\n, dodaje nowego axis, a vstack dodaje poprostu do siebie")

#zad10
print("zad10")
r1 = np.asarray(range(0,24)).reshape((6,4))
wyniki = as_strided(r1, shape=(2, 2, 2, 3), strides=(48, 12, 24, 4))
print(wyniki)
for i in range(2):
    for j in range(2):
        print(np.max(wyniki[i, j]))



