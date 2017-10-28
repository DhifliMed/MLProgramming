from .models import tweet
from django.http import HttpResponse
from .mylib import sentTostem
import random
import numpy as np
from nltk.stem.isri import ISRIStemmer
from sklearn import metrics
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC,SVC
from sklearn.naive_bayes import MultinomialNB,BernoulliNB
import matplotlib.pyplot as pl
from matplotlib.backends.backend_agg import FigureCanvasAgg
import matplotlib.patches as mpatches
class kalimat:
    w=[]
    p=[]
    def new(self):
        self.w=[]
        self.p=[]


def knnimp01():
    dataset = tweet.objects.all()
    knn = KNeighborsClassifier()
    vect = CountVectorizer()
    for d in dataset:
        kalimat.w.append(d.preproc)
        kalimat.p.append(d.polarity)
    kalimatw = vect.fit_transform(kalimat.w)
    acc = np.average(cross_val_score(knn, kalimatw, kalimat.p, cv=10))

    return acc


def knnimp02():
    knn = KNeighborsClassifier()
    vect = TfidfVectorizer()
    kalimatw = vect.fit_transform(kalimat.w)
    s=cross_val_score(knn, kalimatw, kalimat.p, cv=10)
    acc = np.average(s)
    return acc


def knnimp03():
    knn = KNeighborsClassifier(n_neighbors=35)
    vect = TfidfVectorizer()
    kalimatw=vect.fit_transform(kalimat.w)
    acc = np.average(cross_val_score(knn, kalimatw, kalimat.p, cv=10))

    return acc

def knnimp04():
    kalimat.w=[]
    kalimat.p=[]
    dataset = tweet.objects.all()
    for d in dataset:
        kalimat.w.append(sentTostem(d.preproc))
        kalimat.p.append(d.polarity)
    knn = KNeighborsClassifier(n_neighbors=35)
    vect = TfidfVectorizer()
    kalimatw = vect.fit_transform(kalimat.w)
    acc = np.average(cross_val_score(knn, kalimatw, kalimat.p, cv=10))

    return acc
def cmpalgknn():
    scoresknn1 = knnimp01()
    scoresknn2 = knnimp02()
    scoresknn3 = knnimp03()
    scoresknn4 = knnimp04()
    alg = ["-", "Count\nVectorizer", "Tfidf\nVectorizer", "optimisation", "racination", "-"]

    s = []
    s.append(0)
    s.append(scoresknn1)
    s.append(scoresknn2)
    s.append(scoresknn3)
    s.append(scoresknn4)
    s.append(0)
    fig, ax = pl.subplots()

    red_patch = mpatches.Patch(color='r', label='{0:.3f}'.format(round(scoresknn1, 3)))
    green_patch = mpatches.Patch(color='g', label='{0:.3f}'.format(round(scoresknn2, 3)))
    blue_patch = mpatches.Patch(color='b', label='{0:.3f}'.format(round(scoresknn3, 3)))
    yellow_patch = mpatches.Patch(color='y', label='{0:.3f}'.format(round(scoresknn4, 3)))
    pl.legend(handles=[red_patch, green_patch, blue_patch,yellow_patch])

    n1, pknn1, pknn2, pknn3, pknn4, n2 = pl.bar([0, 1, 2, 3, 4, 5], s)

    pknn1.set_facecolor('r')
    pknn2.set_facecolor('g')
    pknn3.set_facecolor('b')
    pknn4.set_facecolor('y')

    ax.set_xticks([0, 1, 2, 3, 4, 5])
    ax.set_xticklabels(alg)
    ax.set_ylim([0, 1])
    ax.set_ylabel('précision')
    ax.set_title("evolution de k-nearest neighbors ")

    n1.set_height(0)
    pknn1.set_height(s[1])
    pknn2.set_height(s[2])
    pknn3.set_height(s[3])
    pknn4.set_height(s[4])
    n2.set_height(0)

    fig.canvas.draw_idle()

    canvas = FigureCanvasAgg(fig)
    response = HttpResponse(content_type="Image/png")
    canvas.print_png(response)
    return response