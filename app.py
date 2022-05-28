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
    flash(d1[0])
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