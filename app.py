from flask import Flask, render_template, request, redirect, session, jsonify
import pandas as pd
import nltk
import os
import json
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
app.secret_key = "chave"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "data", "maquiagens.csv")
USERS_PATH = os.path.join(BASE_DIR, "data", "usuarios.json")

def carregar_dataframe():
    try:
        if not os.path.exists(CSV_PATH):
            raise FileNotFoundError(f"Arquivo não encontrado: {CSV_PATH}")

        df = pd.read_csv(CSV_PATH, encoding="utf-8")
        colunas = ["id", "nome", "categoria", "tom_pele", "marca", "preco", "descricao", "imagem"]
        for col in colunas:
            if col not in df.columns:
                df[col] = ""

        df["id"] = df["id"].astype(int)
        df["preco"] = pd.to_numeric(df["preco"], errors="coerce").fillna(0)

        print(f"CSV carregado: {len(df)} produtos")
        return df

    except Exception as e:
        print(f"ERRO ao carregar CSV: {e}")
        return pd.DataFrame(columns=["id", "nome"])

df = carregar_dataframe()

def carregar_usuarios():
    if not os.path.exists(USERS_PATH):
        return {}
    with open(USERS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def salvar_usuarios(data):
    os.makedirs(os.path.dirname(USERS_PATH), exist_ok=True)
    with open(USERS_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

try:
    nltk.download("stopwords", quiet=True)
    stop_pt = stopwords.words("portuguese")
except:
    stop_pt = ["de", "a", "o", "que", "e"]

df["texto"] = (
    df["nome"].fillna("") + " "
    + df["descricao"].fillna("") + " "
    + df["categoria"].fillna("") + " "
    + df["marca"].fillna("")
)

if df["texto"].str.strip().eq("").all():
    df["texto"] = "produto"

try:
    vectorizer = TfidfVectorizer(stop_words=stop_pt)
    matriz = vectorizer.fit_transform(df["texto"])
except:
    from sklearn.feature_extraction.text import CountVectorizer
    vectorizer = CountVectorizer(stop_words=stop_pt)
    matriz = vectorizer.fit_transform(df["texto"])

def recomendar(pid, n=6):
    try:
        idx = df.index[df["id"] == pid][0]
        similaridade = cosine_similarity(matriz[idx], matriz).flatten()
        indices = similaridade.argsort()[::-1]
        indices = [i for i in indices if i != idx]
        return df.iloc[indices[:n]].to_dict(orient="records")
    except:
        return []

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form["email"]
        senha = request.form["senha"]

        usuarios = carregar_usuarios()
        if email not in usuarios:
            return render_template("login.html", erro="Conta não encontrada")

        user = usuarios[email]
        if isinstance(user, dict):
            if user["senha"] != senha:
                return render_template("login.html", erro="Senha incorreta")
        else:
            if user != senha:
                return render_template("login.html", erro="Senha incorreta")

        session["user"] = email
        return redirect("/catalogo")

    return render_template("login.html")

@app.route("/cadastro", methods=["GET", "POST"])
def cadastro():
    if request.method == "POST":
        email = request.form["email"]
        senha = request.form["senha"]
        nome = request.form.get("nome_completo", "")

        usuarios = carregar_usuarios()
        if email in usuarios:
            return render_template("cadastro.html", erro="Email já cadastrado")

        usuarios[email] = {
            "nome": nome,
            "senha": senha,
            "data_cadastro": str(pd.Timestamp.now())
        }

        salvar_usuarios(usuarios)
        return redirect("/login")

    return render_template("cadastro.html")

@app.route("/api/cadastro", methods=["POST"])
def api_cadastro():
    data = request.get_json()
    if not data:
        return jsonify({"erro": "Dados inválidos"}), 400

    email = data.get("email")
    senha = data.get("senha")
    nome = data.get("nome") or data.get("nome_completo", "")

    if not email or not senha:
        return jsonify({"erro": "Email e senha são obrigatórios"}), 400

    usuarios = carregar_usuarios()
    if email in usuarios:
        return jsonify({"erro": "Email já cadastrado"}), 400

    usuarios[email] = {
        "nome": nome,
        "senha": senha,
        "data_cadastro": str(pd.Timestamp.now())
    }
    salvar_usuarios(usuarios)
    return jsonify({"mensagem": "Conta criada com sucesso!"})

@app.route("/catalogo")
def catalogo():
    return render_template("catalogo.html")

@app.route("/api/maquiagens")
def api_maquiagens():
    produtos = df.to_dict("records")
    return jsonify(produtos)

@app.route("/produto/<int:pid>")
def produto(pid):
    try:
        produto = df[df["id"] == pid].iloc[0].to_dict()
        recs = recomendar(pid)
        return render_template("produto.html", produto=produto, recomendados=recs)
    except:
        return "Produto não encontrado", 404

@app.route("/api/recomendar/<int:pid>")
def api_recomendar(pid):
    return jsonify(recomendar(pid))

def init_cart():
    if "cart" not in session:
        session["cart"] = []
    return session["cart"]

@app.route("/carrinho")
def carrinho_page():
    cart = init_cart()
    total_itens = sum(item['quantidade'] for item in cart)
    total_preco = sum(item['preco'] * item['quantidade'] for item in cart)
    return render_template("carrinho.html", itens=cart, total_itens=total_itens, total_preco=total_preco)

@app.route("/api/carrinho")
def carrinho_listar():
    cart = init_cart()
    total_itens = sum(item['quantidade'] for item in cart)
    total_preco = sum(item['preco'] * item['quantidade'] for item in cart)
    return jsonify({"itens": cart, "total_itens": total_itens, "total_preco": total_preco})

@app.route("/api/carrinho/adicionar/<int:pid>", methods=["POST"])
def carrinho_adicionar(pid):
    cart = init_cart()
    produto = df[df["id"] == pid]
    if produto.empty:
        return jsonify({"erro": "Produto não encontrado"}), 404
    dados = produto.iloc[0].to_dict()
    for item in cart:
        if item["id"] == pid:
            item["quantidade"] += 1
            session.modified = True
            return jsonify({"mensagem": "Quantidade atualizada"})
    cart.append({
        "id": dados["id"],
        "nome": dados["nome"],
        "marca": dados["marca"],
        "preco": float(dados["preco"]),
        "imagem": dados.get("imagem", ""),
        "quantidade": 1
    })
    session.modified = True
    return jsonify({"mensagem": "Adicionado ao carrinho!"})

@app.route("/api/carrinho/atualizar/<int:pid>", methods=["POST"])
def carrinho_atualizar(pid):
    nova_qtd = request.json.get("quantidade")
    cart = init_cart()
    for item in cart:
        if item["id"] == pid:
            item["quantidade"] = max(1, int(nova_qtd))
            session.modified = True
            return jsonify({"mensagem": "Quantidade atualizada"})
    return jsonify({"erro": "Item não encontrado"}), 404

@app.route("/api/carrinho/remover/<int:pid>", methods=["POST"])
def carrinho_remover(pid):
    cart = init_cart()
    cart[:] = [item for item in cart if item["id"] != pid]
    session.modified = True
    return jsonify({"mensagem": "Item removido"})

@app.route("/api/carrinho/limpar", methods=["POST"])
def carrinho_limpar():
    session["cart"] = []
    session.modified = True
    return jsonify({"mensagem": "Carrinho limpo"})

@app.route("/logout")
def logout():
    session.clear()
    return redirect("/")

@app.route("/debug")
def debug():
    return jsonify({
        "produtos": len(df),
        "colunas": df.columns.tolist(),
        "categorias": df["categoria"].unique().tolist(),
        "marcas": df["marca"].unique().tolist()
    })

if __name__ == "__main__":
    app.run(debug=True)