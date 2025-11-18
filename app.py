from flask import Flask, render_template, request, redirect, session, jsonify
import pandas as pd
import nltk
import os
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
app.secret_key = "chave"

# ---------------------------------------------
# CARREGAR CSV (DENTRO DA PASTA /data)
# ---------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "data", "maquiagens.csv")

if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(f"ERRO: Não encontrei o arquivo {CSV_PATH}")

df = pd.read_csv(CSV_PATH, encoding="utf-8")

# ---------------------------------------------
# MACHINE LEARNING (TF-IDF + COSINE)
# ---------------------------------------------
nltk.download("stopwords")
stop_pt = stopwords.words("portuguese")

df["texto"] = (
    df["nome"].fillna("") + " " +
    df["descricao"].fillna("") + " " +
    df["categoria"].fillna("") + " " +
    df["marca"].fillna("")
)

tfidf = TfidfVectorizer(stop_words=stop_pt)
matriz = tfidf.fit_transform(df["texto"])


def recomendar(produto_id, n=6):
    try:
        index = df.index[df["id"] == produto_id][0]
    except:
        return []

    similar = cosine_similarity(matriz[index], matriz).flatten()
    indices = similar.argsort()[::-1]
    indices = [i for i in indices if i != index]

    return df.iloc[indices][:n].to_dict(orient="records")


# ---------------------------------------------
# ROTAS DO SITE
# ---------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        session["user"] = request.form["email"]
        return redirect("/catalogo")
    return render_template("login.html")


@app.route("/catalogo")
def catalogo():
    return render_template("catalogo.html")


@app.route("/produto/<int:pid>")
def produto(pid):
    produto = df[df["id"] == pid].iloc[0]
    recs = recomendar(pid)
    return render_template("produto.html", produto=produto, recomendados=recs)


# ---------------------------------------------
# API para o Catálogo
# ---------------------------------------------
@app.route("/api/maquiagens")
def api_maquiagens():
    return jsonify(df.to_dict(orient="records"))


@app.route("/api/recomendar/<int:pid>")
def api_recomendar(pid):
    return jsonify(recomendar(pid))


@app.route("/logout")
def logout():
    session.clear()
    return redirect("/")


# ---------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)
