import pandas
import pandas as pd
import datetime
import numpy as np
import scipy.stats as sp
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerLine2D
from scipy import stats

def czesc1():
    start = datetime.datetime.strptime("2020-03-01", "%Y-%m-%d")
    end = datetime.datetime.strptime("2020-03-05", "%Y-%m-%d")
    date_generated = pd.date_range(start, end)

    df = pd.DataFrame({
        'data' : date_generated,
        'A' : np.random.normal(size=5),
        'B' : np.random.normal(size=5),
        'C' : np.random.normal(size=5)
    })
    print(df)

    df1 = pd.DataFrame({

        'A': np.random.random(size=20),
        'B': np.random.random(size=20),
        'C': np.random.random(size=20)
    })
    df1.index.name = "index"
    print(df1)
    print(df1.head(3))
    print(df1.tail(3))
    print(df1.index.name)
    print(df1.columns)
    print(df1.values)
    print(df1.sample(5))
    print(df1[['A','B']])
    print(df1.iloc[:3])
    print(df1.iloc[5])
    print(df1.iloc[[0,5,6,7], [1, 2]])
    print(df1.describe(include= [np.number])>0)
    print(df1[df1>0])
    print(df1[df1["A"]>0])
    df1.mean(axis = 0)
    df1.mean(axis = 1)
    df2 = pd.DataFrame({

        'A': np.random.random(size=20),
        'B': np.random.random(size=20),
        'C': np.random.random(size=20)
    })
    df3 = pd.DataFrame({

        'A': np.random.random(size=20),
        'B': np.random.random(size=20),
        'C': np.random.random(size=20)
    })
    zetzet1 = np.transpose(pandas.concat([df2,df3]))
    print(zetzet1)

    df = pd.DataFrame({"x": [1, 2, 3, 4, 5], "y": ['a','b','a','b','b']},
                    index = np.arange(5))
    df.index.name ='id'
    print(df)
    print(df.sort_values("y"))
    print(df.sort_index(ascending=False))

    df = pd.DataFrame(np.random.randn(20, 3), index=np.arange(20), columns=['A', 'B', 'C'])
    df.index.name ='id'
    print(df)
    df['B']=1
    print(df)
    df.iloc[1, 2] = 10
    print(df)
    df[df < 0] = -df
    print(df)
    print("df['B']=1 ustawia cala kolumne B na 1")
    print(" df.iloc[1, 2] = 10 ustawia 1wiersz,2 kolumne na wartosc = 10")
    print("df[df < 0] = -df dla wartosci mniejszych od 0 ustawia przeciwne wartosci")
def zadanie1():
    print("zadanie1:")
    df = pd.DataFrame({"x": [1, 2, 3, 4, 5], 'y': ['a', 'b', 'a', 'b', 'b']})
    g =df.groupby(['y']).mean()
    print(g)

def zadanie2():
    df = pd.DataFrame({"x": [1, 2, 3, 4, 5], 'y': ['a', 'b', 'a', 'b', 'b']})
    g = df.value_counts()
    f = df['y'].value_counts()
    print("wzgledem x\n",g)
    print(f'wzgledem y\n{f}')
def zadanie3():
    df = pd.read_csv('autos.csv')
    #np.loadtxt('autos.csv')
    print("loadtxt nie obsluguje rozszerzenia .csv\n")
def zadanie4():
    df = pd.read_csv('autos.csv')
    g = df.groupby('make')[['city-mpg', 'highway-mpg']].mean()
    print("zad4\n", g)
def zadanie5():
    df = pd.read_csv('autos.csv')
    print(df.groupby('make')['fuel-type'].value_counts())

def zadanie6():
    df = pd.read_csv('autos.csv')
    g = df.loc[:, 'length']
    f = df.loc[:,'city-mpg']
    w1 = np.polyfit(f,g,1)
    w2 = np.polyfit(f,g,2)
    print(w1)
    print(w2)
def zadanie7():
    df = pd.read_csv('autos.csv')
    g = df.loc[:, 'length']
    f = df.loc[:, 'city-mpg']
    print(sp.linregress(f,g))

def zadanie8():
    df = pd.read_csv('autos.csv')
    g = df.loc[:, 'length']
    f = df.loc[:, 'city-mpg']
    a, b = np.polyfit(g, f, 1)
    plt.scatter(g, f)
    plt.plot(g,a*g+b,'red')
    plt.show()



def zadanie9():
    df = pd.read_csv('autos.csv')
    g = df.loc[:, 'length']
    kde = stats.gaussian_kde(g)
    xs = np.linspace(np.min(g),np.max(g))
    y1 = kde(xs)
    fig, ax = plt.subplots()
    ax.hist(g, label = 'probki')
    az = ax.twinx()
    az.plot(xs, y1, label = 'f.gestosci', c = 'r')

    fig.legend()
    plt.show()
def zadanie10():
    df = pd.read_csv('autos.csv')
    g = df.loc[:, 'length']
    f = df.loc[:, 'width']

    kde = stats.gaussian_kde(g)
    xs = np.linspace(np.min(g), np.max(g))
    y1 = kde(xs)

    kde_width = stats.gaussian_kde(f)
    xs_width = np.linspace(np.min(f), np.max(f))
    y_width = kde_width(xs_width)

    fig, (ax_width,ax) = plt.subplots(2)

    ax_width.hist(f, label = 'probki width',color = 'orange')
    az_width = ax_width.twinx()
    az_width.plot(xs_width,y_width,label = 'f.gestosci width ',c = 'g')


    ax.hist(g, label='probki')
    az = ax.twinx()
    az.plot(xs, y1, label='f.gestosci', c='r')
    fig.legend()
    plt.show()

def zadanie11():
    df = pd.read_csv('autos.csv')
    g = df.loc[:, 'length']
    f = df.loc[:, 'width']

    xmin = np.min(g)
    xmax = np.max(g)
    ymin = np.min(f)
    ymax = np.max(f)
    X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    positions = np.vstack([X.ravel(), Y.ravel()])
    values = np.vstack([g, f])
    kernel = stats.gaussian_kde(values)
    Z = np.reshape(kernel(positions).T, X.shape)
    fig, ax = plt.subplots()

    ax.contour((Z).T, cmap=plt.cm.gist_earth_r,
              extent=[xmin, xmax, ymin, ymax])

    ax.plot(g, f, 'k.', markersize=6)
    ax.set_xlim([xmin, xmax])

    ax.set_ylim([ymin, ymax])
    plt.savefig('plot1.png')
    plt.savefig('plot1.pdf')
    plt.show()
czesc1()
zadanie1()
zadanie2()
zadanie3()
zadanie4()
zadanie5()
zadanie6()
zadanie7()
zadanie8()
zadanie9()
zadanie10()
zadanie11()



