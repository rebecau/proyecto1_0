#*********************************************************************************************************************
#*******************************************     PAGINA    ***********************************************************
#*********************************************************************************************************************
from flask import Flask, render_template, request, flash

app = Flask(__name__)
app.secret_key = "1234"

@app.route("/home")
def index():
    flash("What's your name?")
    return render_template("index.html")

@app.route("/pagina1", methods=["POST","GET"])
def pagina1():
    #flash("Hi "+ str(request.form["name_input"]) + ", great to see you!")
    flash(df[0])
    request.form["name_input"]
    return render_template("index.html")

#*********************************************************************************************************************
#*******************************************     CODIGO    ***********************************************************
#*********************************************************************************************************************
import re, collections
import numpy as np ####
import nltk ####
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
nltk.download('stopwords')
import pandas as pd ####

#url = "https://raw.githubusercontent.com/rebecau/ML_PROYECTO1/main/%5BUCI%5D%20AAAI-14%20Accepted%20Papers%20-%20Papers.csv"
url = "https://raw.githubusercontent.com/rebecau/ML_PROYECTO1/main/%5BUCI%5D%20AAAI-14%20Accepted%20Papers%20-%20Papers.csv"
df = pd.read_csv(url)

title = df.iloc[:2, 0]
keyword = df.iloc[:2, 3]
abstract = df.iloc[:2, 5]

#NLP
def Normalizacion(Documentos):
  for i in range(len(Documentos)):#NORMALIZACION ELIMINACION DE CARACTERES ESPECIALES Y MAYUSCULA
    Documentos[i] = re.sub('[^A-Za-z0-9]+', ' ', Documentos[i])
    Documentos[i] = Documentos[i].lower()
  return Documentos

d1 = Normalizacion(title)
d2 = Normalizacion(keyword)
d3 = Normalizacion(abstract)

def TOKENIZACION(Documentos):
  for i in range(len(Documentos)):#TOKENIZACION
    Documentos[i] = Documentos[i].split()
  return Documentos

d1 = TOKENIZACION(d1)
d2 = TOKENIZACION(d2)
d3 = TOKENIZACION(d3)

#STOPWORDS
def STOPWORDS(Documentos):
  for i in range(len(Documentos)):#STOPWORDS
    n = stopwords.words("english")
    for word in Documentos[i]:
      if word in n:
        Documentos[i].remove(word)
  return Documentos

d1 = STOPWORDS(d1)
d2 = STOPWORDS(d2)
d3 = STOPWORDS(d3)

#STEMMING
def STEMMING(Documentos):
  for i in range(len(Documentos)):#stemming(reduce hacia la raiz/ORIGEN)
    stemmer = PorterStemmer()
    stem = Documentos[i]
    for j in range(len(Documentos[i])):
      d11 = stemmer.stem(stem[j])
      stem[j] = d11
    Documentos[i] = stem
  return Documentos

d1 = STEMMING(d1)
d2 = STEMMING(d2)
d3 = STEMMING(d3)

#COSENO
def Coseno(Dic_d3):
  Diccionario3 = []
  for i in range(len(Dic_d3)):
    wordset = Dic_d3[i]
    for j in range(len(wordset)):
      Diccionario3.append(wordset[j])
  Diccionario3 = list(set(Diccionario3))
  return Diccionario3

coseno_d1 = Coseno(d1)
coseno_d2 = Coseno(d2)
coseno_d3 = Coseno(d3)


#Bolsa de palabras (Bag Of Words) -> Modelo de incidencia binaria [Matriz]
def calculateBOW(wordset,l_doc):
  tf_diz = dict.fromkeys(wordset,0)
  bag_words = []
  for i in range(len(l_doc)):
    #print(l_doc[i])#398 resumenes
    #print(len(l_doc[i]))
    cont = []
    vec = l_doc[i]
    for word in wordset:#numero de palabras por resumen
      #print(word," ",l_doc[i])
      tf_diz[word]=l_doc[i].count(word)
      cont.append(vec.count(word))
    bag_words.append(cont)
  return bag_words

bag_words_d1 = calculateBOW(coseno_d1,d1)
bag_words_d2 = calculateBOW(coseno_d2,d2)
bag_words_d3 = calculateBOW(coseno_d3,d3)
bag_words_d1 = np.array(bag_words_d1).reshape(len(d1),len(coseno_d1))
bag_words_d2 = np.array(bag_words_d2).reshape(len(d2),len(coseno_d2))
bag_words_d3 = np.array(bag_words_d3).reshape(len(d3),len(coseno_d3))

uniqueWords_d1 = set(coseno_d1)
uniqueWords_d2 = set(coseno_d2)
uniqueWords_d3 = set(coseno_d3)

numOfWords_d1 = dict.fromkeys(uniqueWords_d1, 0)
numOfWords_d2 = dict.fromkeys(uniqueWords_d2, 0)
numOfWords_d3 = dict.fromkeys(uniqueWords_d3, 0)

def prev_wtf(documentos,numOfWords,uniqueWords):
  num = []
  for i in range(len(documentos)):
    cont = 1
    for word in documentos[i]:
      if cont == 1:
        numOfWords = dict.fromkeys(uniqueWords, 0)
      else:
        numOfWords[word] += 1
        #print(i," [",cont,"/",len(documentos[i]),"] ",word," ",numOfWords[word])
      cont += 1
    num.append(numOfWords)
  return num

numOfWords_d1 = prev_wtf(d1,numOfWords_d1,uniqueWords_d1)
numOfWords_d2 = prev_wtf(d2,numOfWords_d2,uniqueWords_d2)
numOfWords_d3 = prev_wtf(d3,numOfWords_d3,uniqueWords_d3)

def computeTF(wordDict, bagOfWords):
    tfDict = {}
    tf = []
    bagOfWordsCount = len(bagOfWords)
    for i in range(len(bagOfWords)):
      for word, count in wordDict[i].items():
          tfDict[word] = count / float(bagOfWordsCount)
      tf.append(tfDict)
    return tf

tf_d1 = computeTF(numOfWords_d1, d1)
tf_d2 = computeTF(numOfWords_d2, d2)
tf_d3 = computeTF(numOfWords_d3, d3)

def computeIDF(documents):
  import math
  N = len(documents) #numero de documentos
  idfs = []
  idfDict = dict.fromkeys(documents[0].keys(), 0)
  for document in documents:
    for word, val in document.items():
      if val > 0:
        idfDict[word] += 1
    idfs.append(idfDict)
  for word, val in idfDict.items():
    idfDict[word] = math.log(N / (float(val)+1))
  return idfs
    
idfs_d1 = computeIDF(numOfWords_d1)
idfs_d2 = computeIDF(numOfWords_d2)
idfs_d3 = computeIDF(numOfWords_d3)

def Des_Dic(tf, idfs): #Separamos los dicccionarios (valores de sus palabras)
  tf_v = []
  idfs_v = []
  for i in range(len(idfs)):
    tf_v.append([float(x) for x in list(tf[i].values())])
    idfs_v.append([float(x) for x in list(idfs[i].values())])
  return tf_v, idfs_v

tf_v1, idsf_v1 = Des_Dic(tf_d1, idfs_d1)
tf_v2, idsf_v2 = Des_Dic(tf_d2, idfs_d2)
tf_v3, idsf_v3 = Des_Dic(tf_d3, idfs_d3)

def computeTFIDF(tfBagOfWords, idfs):
  tf_idf = []
  for i in range(len(tfBagOfWords)):
    tf_idf.append([x*y for x,y in zip(tfBagOfWords[i],idfs[i])])
  return tf_idf

tf_idf_d1 = computeTFIDF(tf_v1, idsf_v1)
tf_idf_d2 = computeTFIDF(tf_v2, idsf_v2)
tf_idf_d3 = computeTFIDF(tf_v3, idsf_v3)

df =[]
for i in range(len(tf_idf_d1)):
  df.append(pd.DataFrame([tf_idf_d1[i]]))