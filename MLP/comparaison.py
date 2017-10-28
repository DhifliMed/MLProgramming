from .models import tweet
from django.http import HttpResponse
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
def knnimp():
    dataset=tweet.objects.all()

    for d in dataset:
        kalimat.w.append(d.preproc)
        kalimat.p.append(d.polarity)

    X_train, X_test, y_train, y_test = train_test_split(kalimat.w, kalimat.p)
    vect = TfidfVectorizer()
    vect.fit(X_train)
    X_train_dtm = vect.fit_transform(X_train)
    X_test_dtm = vect.transform(X_test)
    knn=KNeighborsClassifier()
    knn.fit(X_train_dtm, y_train)
    y_pred_class = knn.predict(X_test_dtm)
    acc=metrics.accuracy_score(y_test, y_pred_class)

    return 'knn accuracy='+str(acc)
def svmimp():
    dataset=tweet.objects.all()

    for d in dataset:
        kalimat.w.append(d.preproc)
        kalimat.p.append(d.polarity)
    X_train, X_test, y_train, y_test = train_test_split(kalimat.w, kalimat.p)
    vect = CountVectorizer()
    vect.fit(X_train)
    X_train_dtm = vect.transform(X_train)
    X_train_dtm = vect.fit_transform(X_train)
    X_test_dtm = vect.transform(X_test)
    svm=LinearSVC()
    svm.fit(X_train_dtm, y_train)
    y_pred_class = svm.predict(X_test_dtm)
    acc=metrics.accuracy_score(y_test, y_pred_class)
    return 'svm='+str(acc)
def nbimp():

    vect = CountVectorizer()
    dataset=tweet.objects.all()

    nb = MultinomialNB()
    html='nb accuracity = '
    for d in dataset:
        kalimat.w.append(d.preproc)
        kalimat.p.append(d.polarity)

    X_train, X_test, y_train, y_test = train_test_split(kalimat.w, kalimat.p,random_state=0)
    vect.fit(X_train)
    X_train_dtm = vect.fit_transform(X_train)
    X_test_dtm = vect.transform(X_test)
    nb.fit(X_train_dtm, y_train)
    y_pred_class = nb.predict(X_test_dtm)
    acc=metrics.accuracy_score(y_test, y_pred_class)
    html=html+str(acc)
    return html
def nbimp2():
    acc = []
    iaxes=[]
    jaxes=[]
    l = 10
    j = 0
    vect = CountVectorizer()
    dataset=tweet.objects.all()
    nb = MultinomialNB()
    html='nb accuracity = '
    for d in dataset:
        kalimat.w.append(d.preproc)
        kalimat.p.append(d.polarity)

    for i in range(l):
        j=j+1/l
        iaxes.append(j*100)
        X_train, X_test, y_train, y_test = train_test_split(kalimat.w, kalimat.p,train_size=j,random_state=0)
        vect.fit(X_train)
        X_train_dtm = vect.fit_transform(X_train)
        X_test_dtm = vect.transform(X_test)
        nb.fit(X_train_dtm, y_train)
        y_pred_class = nb.predict(X_test_dtm)
        acc.append(metrics.accuracy_score(y_test, y_pred_class))
        jaxes.append(acc[i])
    f = pl.figure()
    pl.plot(iaxes,jaxes)
    pl.xlabel('% echantillon de teste')
    pl.ylabel('accuracitée')
    pl.title("cross validation")
    canvas = FigureCanvasAgg(f)
    response = HttpResponse(content_type='image/png')
    canvas.print_png(response)
    pl.close(f)

    return response
def varkknn():
    #variation de k
    acc = []
    vect = TfidfVectorizer()
    dataset=tweet.objects.all()
    knn = KNeighborsClassifier()
    for d in dataset:
        kalimat.w.append(d.preproc)
        kalimat.p.append(d.polarity)
    kalimatw = vect.fit_transform(kalimat.w)
    s=[]
    l=100
    i=0
    j=0
    max = 0
    while(i<l):
        knn = KNeighborsClassifier(n_neighbors=i+1)
        s.append(np.round(np.max(cross_val_score(knn, kalimatw, kalimat.p, cv=4)),3))
        if(max<s[i]):
            max=s[i]
            j=i
        print(str(i+1)+"% max="+str(j))

    f = pl.figure()
    red_patch = mpatches.Patch(color='black', label='meilleur resultat={0:.3f} pour k='.format(max)+str(j))
    pl.legend(handles=[red_patch])
    pl.plot(acc, 'r')
    pl.axis([0, l, 0, 1])
    pl.xlabel('k')
    pl.ylabel('précision')
    pl.title("comparison de précision par variation de k ")
    canvas = FigureCanvasAgg(f)
    response = HttpResponse(content_type='image/png')
    canvas.print_png(response)


    return response
def svmimp2():
    acc = []
    vect = CountVectorizer()
    dataset=tweet.objects.all()
    svm = LinearSVC()
    html='svm accuracity = '
    for d in dataset:
        kalimat.w.append(d.preproc)
        kalimat.p.append(d.polarity)
    l=30
    j=0
    for i in range(l):
        j=j+1/l
        X_train, X_test, y_train, y_test = train_test_split(kalimat.w, kalimat.p,train_size=j,random_state=0)
        vect.fit(X_train)
        X_train_dtm = vect.fit_transform(X_train)
        X_test_dtm = vect.transform(X_test)
        svm.fit(X_train_dtm, y_train)
        y_pred_class = svm.predict(X_test_dtm)
        acc.append(metrics.accuracy_score(y_test, y_pred_class))
        html=html+str(acc[i])+'<br>'
    return html
def kersvm():
    #comparaison de kernel
    ker = ["-", "linear", "poly", "rbf", "-"]
    vect = TfidfVectorizer()
    dataset=tweet.objects.all()
    for d in dataset:
        kalimat.w.append(d.preproc)
        kalimat.p.append(d.polarity)
    kalimatw = vect.fit_transform(kalimat.w)
    s=[]
    s.append(0)
    for i in range(3):
        svm = SVC(kernel=ker[i+1])
        s.append(np.max(np.round(cross_val_score(svm, kalimatw, kalimat.p, cv=10),3)))
    s.append(0)
    fig, ax = pl.subplots()
    n1,pli, ppol, prbf , n2 = pl.bar([0,1,2,3,4], s)
    red_patch = mpatches.Patch(color='r', label='{0:.3f}'.format(s[1]))
    green_patch = mpatches.Patch(color='g', label='{0:.3f}'.format(s[2]))
    blue_patch = mpatches.Patch(color='b', label='{0:.3f}'.format(s[3]))
    ax.legend(handles=[red_patch, green_patch, blue_patch])

    pli.set_facecolor('r')
    ppol.set_facecolor('g')
    prbf.set_facecolor('b')
    n1.set_height(0)
    pli.set_height(s[1])
    ppol.set_height(s[2])
    prbf.set_height(s[3])
    n2.set_height(0)

    ax.set_xticks([0, 1, 2, 3, 4])
    ax.set_xticklabels(ker)
    ax.set_ylim([0, 1])
    ax.set_ylabel('précision')

    ax.set_title("comparaison de précision par changement de kernel ")
    fig.canvas.draw_idle()
    canvas = FigureCanvasAgg(fig)
    response = HttpResponse(content_type='image/png')
    canvas.print_png(response)
    return response
def knnimp3():
    dataset = tweet.objects.all()

    for d in dataset:
        kalimat.w.append(d.preproc)
        kalimat.p.append(d.polarity)
    knn = KNeighborsClassifier()
    vect = TfidfVectorizer()
    kalimatw = vect.fit_transform(kalimat.w)
    scores = cross_val_score(knn, kalimatw, kalimat.p, cv=10)

    f = pl.figure()
    pl.axis([0, 10, 0, 1])
    pl.plot(scores)
    pl.xlabel('% numero de teste')
    pl.ylabel('accuracitée')
    pl.title("cross validation")
    canvas = FigureCanvasAgg(f)
    response = HttpResponse(content_type='image/png')
    canvas.print_png(response)

    pl.close(f)

    return response
def svmimp3():
    dataset = tweet.objects.all()
    for d in dataset:
        kalimat.w.append(d.preproc)
        kalimat.p.append(d.polarity)
    svm = LinearSVC()

    vect = TfidfVectorizer()
    kalimatw = vect.fit_transform(kalimat.w)
    scores = cross_val_score(svm, kalimatw, kalimat.p, cv=10)

    f = pl.figure()
    pl.axis([0, 10, 0, 1])
    pl.plot(scores)
    pl.xlabel('% numero de teste')
    pl.ylabel('accuracitée')
    pl.title("cross validation")
    canvas = FigureCanvasAgg(f)
    response = HttpResponse(content_type='image/png')
    canvas.print_png(response)

    pl.close(f)

    return response
def nbimp3():
    dataset = tweet.objects.all()

    for d in dataset:
        kalimat.w.append(d.preproc)
        kalimat.p.append(d.polarity)
    nb = MultinomialNB()
    vect = TfidfVectorizer()
    kalimatw = vect.fit_transform(kalimat.w)
    scores = cross_val_score(nb, kalimatw, kalimat.p, cv=10)

    f = pl.figure()
    pl.axis([0, 10, 0, 1])
    pl.plot(scores)
    pl.xlabel('% numero de teste')
    pl.ylabel('accuracitée')
    pl.title("cross validation")
    canvas = FigureCanvasAgg(f)
    response = HttpResponse(content_type='image/png')
    canvas.print_png(response)

    pl.close(f)

    return response
def nbtype():
    s = []
    vect = TfidfVectorizer()
    dataset = tweet.objects.all()

    NBtype = ['', 'MultinomialNB', 'BernoulliNB','']
    for d in dataset:
        kalimat.w.append(d.preproc)
        kalimat.p.append(d.polarity)
    mnb=MultinomialNB()
    bnb=BernoulliNB()
    kalimatw = vect.fit_transform(kalimat.w)
    scoresmnb = cross_val_score(mnb, kalimatw, kalimat.p, cv=10)
    scoresbnb = cross_val_score(bnb, kalimatw, kalimat.p, cv=10)

    s.append(0)
    s.append(np.round(np.max(scoresmnb),3))
    s.append(np.round(np.max(scoresbnb),3))
    s.append(0)

    fig, ax = pl.subplots()
    n1, pmnb, pbnb, n2 = pl.bar([0, 1, 2, 3], s)

    pmnb.set_facecolor('g')
    pbnb.set_facecolor('b')

    green_patch = mpatches.Patch(color='g', label='{0:.3f}'.format(s[1]))
    blue_patch = mpatches.Patch(color='b', label='{0:.3f}'.format(s[2]))

    ax.set_xticks([0, 1, 2, 3])
    ax.set_xticklabels(NBtype)
    ax.set_ylim([0, 1])
    ax.set_ylabel('précision')
    ax.set_title("comparison de précision par type de naif bayes ")
    n1.set_height(0)
    pmnb.set_height(s[1])
    pbnb.set_height(s[2])
    n2.set_height(0)
    pl.legend(handles=[green_patch, blue_patch])
    fig.canvas.draw_idle()
    canvas = FigureCanvasAgg(fig)
    response = HttpResponse(content_type='image/png')
    canvas.print_png(response)
    return response
def sentTostem(a):
    b=''
    stem = ISRIStemmer()
    a=stem.waw(a)
    a=stem.suf32(a)
    a=stem.pre32(a)
    a=stem.norm(a)
    c=a.split(" ")
    for w in c:
        b+=str(stem.stem(w))+" "
    return str(b)
def Impalg():
    dataset = tweet.objects.all()
    knn = KNeighborsClassifier(n_neighbors=35)
    svm = SVC(kernel="linear")
    nb = MultinomialNB()
    vect = TfidfVectorizer()
    for d in dataset:
        kalimat.w.append(sentTostem(d.preproc))
        kalimat.p.append(d.polarity)

    kalimatw = vect.fit_transform(kalimat.w)
    scoresknn = cross_val_score(knn, kalimatw, kalimat.p, cv=10)
    scoressvm = cross_val_score(svm, kalimatw, kalimat.p, cv=10)
    scoresnb = cross_val_score(nb, kalimatw, kalimat.p, cv=10)

    f = pl.figure()
    pl.plot(scoresknn,'r')
    pl.plot(scoresnb, 'g')
    pl.plot(scoressvm, 'b')
    pl.axis([0, 9, 0, 1])

    red_patch = mpatches.Patch(color='r', label='knn={0:.3f}'.format(round(np.max(scoresknn),3)))
    green_patch = mpatches.Patch(color='g', label='nb={0:.3f}'.format(round(np.max(scoresnb),3)))
    blue_patch = mpatches.Patch(color='b', label='svm={0:.3f}'.format(round(np.max(scoressvm),3)))
    pl.legend(title="maximum" ,handles=[red_patch, green_patch, blue_patch])

    #pl.annotate('knn', xy=(2, scoresknn[2]), xytext=(2, scoresknn[2]+0.15),arrowprops=dict(facecolor='red', shrink=0.05),color='red')
    #pl.annotate('svm', xy=(3, scoressvm[3]), xytext=(3, scoressvm[3]+0.15),arrowprops=dict(facecolor='blue', shrink=0.05),color='blue')
    #pl.annotate('nb', xy=(4, scoresnb[4]), xytext=(4, scoresnb[4]+0.15),arrowprops=dict(facecolor='green', shrink=0.05),color='green')
    pl.xlabel('numéro de teste')
    pl.ylabel('précision')
    pl.title("comparaison des algorithmes par cross validation ")
    canvas = FigureCanvasAgg(f)
    response = HttpResponse(content_type='image/png')
    canvas.print_png(response)

    pl.close(f)

    return response
def cmpalg():
    dataset = tweet.objects.all()

    knn = KNeighborsClassifier()
    svm = SVC()
    nb = MultinomialNB()
    vect = CountVectorizer()
    for d in dataset:
        kalimat.w.append(d.preproc)
        kalimat.p.append(d.polarity)

    kalimatw = vect.fit_transform(kalimat.w)
    scoresknn = cross_val_score(knn, kalimatw, kalimat.p, cv=10)
    scoressvm = cross_val_score(svm, kalimatw, kalimat.p, cv=10)
    scoresnb = cross_val_score(nb, kalimatw, kalimat.p, cv=10)
    alg=["-","K nearest \nneighbors","naif bayes","support vector\n machine","-"]


    s=[]
    s.append(0)
    s.append(float(np.average(scoresknn)))
    s.append(float(np.average(scoresnb)))
    s.append(float(np.average(scoressvm)))
    s.append(0)
    fig, ax = pl.subplots()

    red_patch = mpatches.Patch(color='r', label='{0:.3f}'.format(round(np.average(scoresknn),3)))
    green_patch = mpatches.Patch(color='g', label='{0:.3f}'.format(round(np.average(scoresnb),3)))
    blue_patch = mpatches.Patch(color='b', label='{0:.3f}'.format(round(np.average(scoressvm),3)))
    pl.legend(handles=[red_patch, green_patch, blue_patch])

    n1,pknn, pnb, psvm ,n2= pl.bar([0 , 1 , 2 , 3 , 4], s)

    pknn.set_facecolor('r')
    pnb.set_facecolor('g')
    psvm.set_facecolor('b')

    ax.set_xticks([0 , 1 , 2 , 3 , 4])
    ax.set_xticklabels(alg)
    ax.set_ylim([0, 1])
    ax.set_ylabel('précision')
    ax.set_title("comparaison des algorithmes par moyenne de précision ")

    n1.set_height(0)
    pknn.set_height(s[1])
    pnb.set_height(s[2])
    psvm.set_height(s[3])
    n2.set_height(0)

    fig.canvas.draw_idle()

    canvas = FigureCanvasAgg(fig)
    response = HttpResponse(content_type="Image/png")
    canvas.print_png(response)
    return response
def knnimp01():
    dataset = tweet.objects.all()

    for d in dataset:
        kalimat.w.append(d.preproc)
        kalimat.p.append(d.polarity)

    X_train, X_test, y_train, y_test = train_test_split(kalimat.w, kalimat.p)
    vect = CountVectorizer()
    vect.fit(X_train)
    X_train_dtm = vect.fit_transform(X_train)
    X_test_dtm = vect.transform(X_test)
    knn = KNeighborsClassifier()
    knn.fit(X_train_dtm, y_train)
    y_pred_class = knn.predict(X_test_dtm)
    acc = metrics.accuracy_score(y_test, y_pred_class)

    return acc
def knnimp02():
    dataset = tweet.objects.all()

    for d in dataset:
        kalimat.w.append(d.preproc)
        kalimat.p.append(d.polarity)

    X_train, X_test, y_train, y_test = train_test_split(kalimat.w, kalimat.p)
    vect = TfidfVectorizer()
    vect.fit(X_train)
    X_train_dtm = vect.fit_transform(X_train)
    X_test_dtm = vect.transform(X_test)
    knn = KNeighborsClassifier()
    knn.fit(X_train_dtm, y_train)
    y_pred_class = knn.predict(X_test_dtm)
    acc = metrics.accuracy_score(y_test, y_pred_class)

    return acc

def knnimp03():
    dataset = tweet.objects.all()

    for d in dataset:
        kalimat.w.append(d.preproc)
        kalimat.p.append(d.polarity)

    X_train, X_test, y_train, y_test = train_test_split(kalimat.w, kalimat.p)
    vect = TfidfVectorizer()
    vect.fit(X_train)
    X_train_dtm = vect.fit_transform(X_train)
    X_test_dtm = vect.transform(X_test)
    knn = KNeighborsClassifier()
    knn.fit(X_train_dtm, y_train)
    y_pred_class = knn.predict(X_test_dtm)
    acc = metrics.accuracy_score(y_test, y_pred_class)

    return acc
def cmpalgknn():

    scoresknn1 = knnimp01()
    scoresknn1 = knnimp02()
    scoresknn1 = knnimp03()
    scoresknn1 = knnimp04()
    alg=["-","CountVectorizer","TfidfVectorizer","optimization","racination","-"]


    s=[]
    s.append(0)
    s.append(float(np.average(scoresknn)))
    s.append(float(np.average(scoresnb)))
    s.append(float(np.average(scoressvm)))
    s.append(0)
    fig, ax = pl.subplots()

    red_patch = mpatches.Patch(color='r', label='{0:.3f}'.format(round(np.average(scoresknn),3)))
    green_patch = mpatches.Patch(color='g', label='{0:.3f}'.format(round(np.average(scoresnb),3)))
    blue_patch = mpatches.Patch(color='b', label='{0:.3f}'.format(round(np.average(scoressvm),3)))
    pl.legend(handles=[red_patch, green_patch, blue_patch])

    n1,pknn, pnb, psvm ,n2= pl.bar([0 , 1 , 2 , 3 , 4], s)

    pknn.set_facecolor('r')
    pnb.set_facecolor('g')
    psvm.set_facecolor('b')

    ax.set_xticks([0 , 1 , 2 , 3 , 4])
    ax.set_xticklabels(alg)
    ax.set_ylim([0, 1])
    ax.set_ylabel('précision')
    ax.set_title("comparaison des algorithmes par moyenne de précision ")

    n1.set_height(0)
    pknn.set_height(s[1])
    pnb.set_height(s[2])
    psvm.set_height(s[3])
    n2.set_height(0)

    fig.canvas.draw_idle()

    canvas = FigureCanvasAgg(fig)
    response = HttpResponse(content_type="Image/png")
    canvas.print_png(response)
    return response






def importdata():
    import pandas as pd
    t=tweet()
    data = pd.read_json('C:\\Users\\DToshiba\\Documents\\GitHub\\DhifliHammami\\MLProgramming\\MLProgramming\\MLProgramming\\MLP\\static\\ASTD_pos_neg_neutral_preprocessed.json')
    for index, row in data.iterrows():
        tweets.t.append(row.text)
        tweets.w.append(row.preprocessed)
        tweets.p.append(row.polarity)
    aux = list(zip(tweets.w, tweets.t,tweets.p))
    random.shuffle(aux)
    tweets.w, tweets.t , tweets.p = zip(*aux)
    for i in range(len(tweets.w)):
        t = tweet(text=tweets.t[i],preproc=tweets.w[i],polarity=tweets.p[i])
        t.save()