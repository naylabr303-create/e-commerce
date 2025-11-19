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

# ---------------------------------------------
# CARREGAR CSV (DENTRO DA PASTA /data) - CORRIGIDO
# ---------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "data", "maquiagens.csv")

if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(f"ERRO: Não encontrei o arquivo {CSV_PATH}")

# CORREÇÃO: Lidar com possíveis erros no CSV
try:
    # Tenta ler com diferentes abordagens
    df = pd.read_csv(CSV_PATH, encoding="utf-8", on_bad_lines='skip', engine='python')
except Exception as e:
    print(f"Erro ao ler CSV: {e}")
    # Tenta alternativa se o método acima falhar
    try:
        df = pd.read_csv(CSV_PATH, encoding="utf-8", error_bad_lines=False)
    except:
        # Última tentativa com encoding diferente
        df = pd.read_csv(CSV_PATH, encoding="latin-1", on_bad_lines='skip', engine='python')

# Verifica se o DataFrame foi carregado corretamente
if df.empty:
    raise ValueError("DataFrame vazio - verifique o arquivo CSV")

print(f"CSV carregado com {len(df)} linhas e {len(df.columns)} colunas")
print(f"Colunas: {df.columns.tolist()}")

# ---------------------------------------------
# ARQUIVO DE USUÁRIOS JSON
# ---------------------------------------------
USER_FILE = os.path.join("data", "usuarios.json")

def carregar_usuarios():
    if not os.path.exists(USER_FILE):
        return {}
    with open(USER_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def salvar_usuarios(usuarios):
    # Garante que o diretório existe
    os.makedirs(os.path.dirname(USER_FILE), exist_ok=True)
    with open(USER_FILE, "w", encoding="utf-8") as f:
        json.dump(usuarios, f, ensure_ascii=False, indent=4)

# ---------------------------------------------
# MACHINE LEARNING (TF-IDF + COSINE)
# ---------------------------------------------
try:
    nltk.download("stopwords", quiet=True)
    stop_pt = stopwords.words("portuguese")
except:
    # Fallback para stopwords em português
    stop_pt = ["de", "a", "o", "que", "e", "do", "da", "em", "um", "para", "é", "com", "não", "uma", "os", "no", "se", "na", "por", "mais", "as", "dos", "como", "mas", "foi", "ao", "ele", "das", "tem", "à", "seu", "sua", "ou", "ser", "quando", "muito", "há", "nos", "já", "está", "eu", "também", "só", "pelo", "pela", "até", "isso", "ela", "entre", "era", "depois", "sem", "mesmo", "aos", "ter", "seus", "quem", "nas", "me", "esse", "eles", "estão", "você", "tinha", "foram", "essa", "num", "nem", "suas", "meu", "às", "minha", "têm", "numa", "pelos", "elas", "havia", "seja", "qual", "será", "nós", "tenho", "lhe", "deles", "essas", "esses", "pelas", "este", "fosse", "dele", "tu", "te", "vocês", "vos", "lhes", "meus", "minhas", "teu", "tua", "teus", "tuas", "nosso", "nossa", "nossos", "nossas", "dela", "delas", "esta", "estes", "estas", "aquele", "aquela", "aqueles", "aquelas", "isto", "aquilo"]

# CORREÇÃO: Verifica se as colunas existem antes de criar o texto
colunas_necessarias = ["nome", "descricao", "categoria", "marca"]
for col in colunas_necessarias:
    if col not in df.columns:
        print(f"Aviso: Coluna '{col}' não encontrada no DataFrame")
        df[col] = ""  # Cria coluna vazia se não existir

# Garante que a coluna 'id' existe
if "id" not in df.columns:
    print("Aviso: Criando coluna 'id' automática")
    df["id"] = range(1, len(df) + 1)

df["texto"] = (
    df["nome"].fillna("") + " " +
    df["descricao"].fillna("") + " " +
    df["categoria"].fillna("") + " " +
    df["marca"].fillna("")
)

# CORREÇÃO: Verifica se há texto para processar
if df["texto"].str.strip().eq("").all():
    print("Aviso: Coluna 'texto' está vazia")
    df["texto"] = "produto"  # Valor padrão

try:
    tfidf = TfidfVectorizer(stop_words=stop_pt)
    matriz = tfidf.fit_transform(df["texto"])
except Exception as e:
    print(f"Erro no TF-IDF: {e}")
    # Fallback simples
    from sklearn.feature_extraction.text import CountVectorizer
    tfidf = CountVectorizer(stop_words=stop_pt)
    matriz = tfidf.fit_transform(df["texto"])

def recomendar(produto_id, n=6):
    try:
        index = df.index[df["id"] == produto_id][0]
    except IndexError:
        print(f"Produto ID {produto_id} não encontrado")
        return []
    except Exception as e:
        print(f"Erro ao buscar produto: {e}")
        return []

    try:
        similar = cosine_similarity(matriz[index], matriz).flatten()
        indices = similar.argsort()[::-1]
        indices = [i for i in indices if i != index]

        return df.iloc[indices][:n].to_dict(orient="records")
    except Exception as e:
        print(f"Erro na recomendação: {e}")
        return []

# ---------------------------------------------
# ROTA API DE CADASTRO (USADA PELO SEU FORM JS)
# ---------------------------------------------
@app.route("/api/cadastro", methods=["POST"])
def api_cadastro():
    data = request.get_json()

    nome = data.get("nome")
    email = data.get("email")
    senha = data.get("senha")

    if not nome or not email or not senha:
        return jsonify({"erro": "Preencha todos os campos!"}), 400

    usuarios = carregar_usuarios()

    if email in usuarios:
        return jsonify({"erro": "Este email já está cadastrado!"}), 400

    usuarios[email] = senha
    salvar_usuarios(usuarios)

    return jsonify({"mensagem": "Conta criada com sucesso!"}), 200

# ---------------------------------------------
# ROTAS NORMAIS (TELAS)
# ---------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")

# -----------------------
# LOGIN
# -----------------------
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form["email"]
        senha = request.form["senha"]

        usuarios = carregar_usuarios()

        if email not in usuarios:
            return render_template("login.html", erro="Conta não encontrada. Cadastre-se!")

        if usuarios[email] != senha:
            return render_template("login.html", erro="Senha incorreta!")

        session["user"] = email
        return redirect("/catalogo")

    return render_template("login.html")

# -----------------------
# CADASTRO NORMAL (NÃO USADO PELO HTML ENVIADO)
# -----------------------
@app.route("/cadastro", methods=["GET", "POST"])
def cadastro():
    if request.method == "POST":
        email = request.form["email"]
        senha = request.form["senha"]

        usuarios = carregar_usuarios()

        if email in usuarios:
            return render_template("cadastro.html", erro="Este email já está cadastrado!")

        usuarios[email] = senha
        salvar_usuarios(usuarios)

        return redirect("/login")

    return render_template("cadastro.html")

# -----------------------
# CATÁLOGO
# -----------------------
@app.route("/catalogo")
def catalogo():
    return render_template("catalogo.html")

# -----------------------
# PÁGINA DO PRODUTO + RECOMENDAÇÕES
# -----------------------
@app.route("/produto/<int:pid>")
def produto(pid):
    try:
        produto = df[df["id"] == pid].iloc[0]
        recs = recomendar(pid)
        return render_template("produto.html", produto=produto, recomendados=recs)
    except IndexError:
        return "Produto não encontrado", 404
    except Exception as e:
        print(f"Erro na página do produto: {e}")
        return "Erro interno", 500

# APIs auxiliares
@app.route("/api/maquiagens")
def api_maquiagens():
    return jsonify(df.to_dict(orient="records"))

@app.route("/api/recomendar/<int:pid>")
def api_recomendar(pid):
    return jsonify(recomendar(pid))

# LOGOUT
@app.route("/logout")
def logout():
    session.clear()
    return redirect("/")

# ---------------------------------------------
# RODAR SERVIDOR
# ---------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)