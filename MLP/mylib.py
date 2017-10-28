from .models import tweet
import numpy as np
from nltk.stem.isri import ISRIStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB,BernoulliNB
class tweets:
    w=[]
    p=[]
    t=[]
class kalimat:
    w=[]
    p=[]
    def new(self):
        self.w=[]
        self.p=[]
def comptermot():
    dataset = tweet.objects.all()
    s=''
    for d in dataset:
        s+= d.preproc+' '
    wc=s.split(" ")
    return (str(len(wc)))
def predictknnp(b,p,s):
    dataset = tweet.objects.all()
    k=kalimat()
    k.new()
    for d in dataset:
        if (s == 0):
            k.w.append(d.preproc)
        else:
            k.w.append(sentTostem(d.preproc))
        k.p.append(d.polarity)
    vect = TfidfVectorizer()
    knn = KNeighborsClassifier(n_neighbors=p)
    knn.fit(vect.fit_transform(k.w), k.p)
    a = []
    a.append(b)
    ap = vect.transform(a)

    c = knn.predict(ap)
    ctxt = ''
    if (c == -1):
        ctxt = 'negative'
    elif (c == 1):
        ctxt = 'positive'
    else:
        ctxt = 'neutral'
    return 'K-Nearest neighbors classe la phrase <<' + b + '>> comme ' + ctxt
def predictnbp(b,p,s):
    dataset=tweet.objects.all()
    k = kalimat()
    k.new()
    for d in dataset:
        if (s == 0):
            k.w.append(d.preproc)
        else:
            k.w.append(sentTostem(d.preproc))
        k.p.append(d.polarity)
    if(p==0):
        nb=MultinomialNB()
    else:
        nb=BernoulliNB()
    vect = TfidfVectorizer()
    nb.fit(vect.fit_transform(k.w), k.p)
    a=[]
    a.append(b)
    ap = vect.transform(a)
    c=nb.predict(ap)
    ctxt = ''
    if(c==-1):
        ctxt='negative'
    elif (c == 1):
        ctxt = 'positive'
    else :
        ctxt = 'neutre'
    return 'Naive Bayes classe la phrase <<' + b + '>> comme ' + ctxt
def predictsvmp(b,p,s):
    dataset=tweet.objects.all()
    k = kalimat()
    k.new()
    for d in dataset:
        if (s == 0):
            k.w.append(d.preproc)
        else:
            k.w.append(sentTostem(d.preproc))
        k.p.append(d.polarity)
    vect = TfidfVectorizer()
    svm = SVC()
    if(p==0):
        svm.kernel="linear"
    elif(p==1):
        svm.kernel = "rbf"
    else:
        svm.kernel = "poly"
    svm.fit(vect.fit_transform(k.w), k.p)
    a=[]
    a.append(b)
    ap = vect.transform(a)
    c=svm.predict(ap)
    ctxt = ''
    if (c == -1):
        ctxt = 'negative'
    elif (c == 1):
        ctxt = 'positive'
    else:
        ctxt = 'neutre'
    return 'Support Vector Machine classe la phrase <<'+b+'>> comme '+ctxt
def predictknn(b):
    dataset = tweet.objects.all()
    k = kalimat()
    k.new()
    for d in dataset:
        k.w.append(d.preproc)
        k.p.append(d.polarity)
    vect = TfidfVectorizer()
    X_train_dtm = vect.fit_transform(kalimat.w)
    knn = KNeighborsClassifier(n_neighbors=35)
    knn.fit(X_train_dtm, k.p)
    a = []
    a.append(b)
    ap = vect.transform(a)

    c = knn.predict(ap)
    ctxt = ''
    if (c == -1):
        ctxt = 'negative'
    elif (c == 1):
        ctxt = 'positive'
    else:
        ctxt = 'neutral'
    return 'K-Nearest neighbors classe la phrase <<' + b + '>> comme ' + ctxt
def predictnb(b):
    dataset=tweet.objects.all()
    k = kalimat()
    k.new()
    for d in dataset:
        k.w.append(d.preproc)
        k.p.append(d.polarity)
    vect = TfidfVectorizer()
    X_train_dtm = vect.fit_transform(k.w)
    nb = MultinomialNB()
    nb.fit(X_train_dtm, k.p)
    a=[]
    a.append(b)
    ap = vect.transform(a)
    c=nb.predict(ap)
    ctxt = ''
    if(c==-1):
        ctxt='negative'
    elif (c == 1):
        ctxt = 'positive'
    else :
        ctxt = 'neutre'
    return 'Naive Bayes classe la phrase <<' + b + '>> comme ' + ctxt
def predictsvm(b):
    dataset=tweet.objects.all()
    k = kalimat()
    k.new()
    for d in dataset:
        k.w.append(d.preproc)
        k.p.append(d.polarity)
    vect = TfidfVectorizer()
    X_train_dtm = vect.fit_transform(k.w)
    svm = SVC()
    svm.kernel="linear"
    svm.fit(X_train_dtm, k.p)
    a=[]
    a.append(b)
    ap = vect.transform(a)
    c=svm.predict(ap)
    ctxt = ''
    if (c == -1):
        ctxt = 'negative'
    elif (c == 1):
        ctxt = 'positive'
    else:
        ctxt = 'neutre'
    return 'Support Vector Machine classe la phrase <<'+b+'>> comme '+ctxt
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