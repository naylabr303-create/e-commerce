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
# CARREGAR CSV - VERSÃO CORRIGIDA (SEM EMOJIS)
# ---------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "data", "maquiagens.csv")

def carregar_dataframe():
    """Carrega o DataFrame com tratamento de erros"""
    try:
        if not os.path.exists(CSV_PATH):
            raise FileNotFoundError(f"ERRO: Arquivo {CSV_PATH} não encontrado")
        
        # Tenta ler o CSV
        df = pd.read_csv(CSV_PATH, encoding='utf-8')
        
        # Verifica se as colunas necessárias existem
        colunas_necessarias = ['id', 'nome', 'categoria', 'tom_pele', 'marca', 'preco']
        for coluna in colunas_necessarias:
            if coluna not in df.columns:
                print(f"AVISO: Coluna '{coluna}' não encontrada no CSV")
                return None
        
        # Limpa dados
        df = df.dropna(subset=['id', 'nome'])  # Remove linhas sem ID ou nome
        df['id'] = df['id'].astype(int)
        df['preco'] = pd.to_numeric(df['preco'], errors='coerce').fillna(0)
        
        print(f"SUCESSO: CSV carregado com {len(df)} produtos")
        print(f"CATEGORIAS: {df['categoria'].unique()}")
        print(f"MARCAS: {df['marca'].unique()}")
        
        return df
        
    except Exception as e:
        print(f"ERRO CRITICO ao carregar CSV: {e}")
        return None

# Carrega o DataFrame
df = carregar_dataframe()

if df is None:
    print("ERRO: Não foi possível carregar o CSV. Criando DataFrame vazio.")
    df = pd.DataFrame(columns=['id', 'nome', 'categoria', 'tom_pele', 'marca', 'preco', 'descricao', 'imagem'])

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
        print(f"AVISO: Coluna '{col}' não encontrada no DataFrame")
        df[col] = ""  # Cria coluna vazia se não existir

# Garante que a coluna 'id' existe
if "id" not in df.columns:
    print("AVISO: Criando coluna 'id' automática")
    df["id"] = range(1, len(df) + 1)

df["texto"] = (
    df["nome"].fillna("") + " " +
    df["descricao"].fillna("") + " " +
    df["categoria"].fillna("") + " " +
    df["marca"].fillna("")
)

# CORREÇÃO: Verifica se há texto para processar
if df["texto"].str.strip().eq("").all():
    print("AVISO: Coluna 'texto' está vazia")
    df["texto"] = "produto"  # Valor padrão

try:
    tfidf = TfidfVectorizer(stop_words=stop_pt)
    matriz = tfidf.fit_transform(df["texto"])
except Exception as e:
    print(f"ERRO no TF-IDF: {e}")
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
        print(f"ERRO ao buscar produto: {e}")
        return []

    try:
        similar = cosine_similarity(matriz[index], matriz).flatten()
        indices = similar.argsort()[::-1]
        indices = [i for i in indices if i != index]

        return df.iloc[indices][:n].to_dict(orient="records")
    except Exception as e:
        print(f"ERRO na recomendação: {e}")
        return []

# ---------------------------------------------
# ROTA DEBUG PARA VERIFICAR DADOS
# ---------------------------------------------
@app.route("/debug")
def debug():
    """Página de debug para verificar os dados"""
    if df.empty:
        info = {
            "status": "ERRO: DataFrame vazio",
            "total_produtos": 0,
            "colunas": [],
            "primeiros_5": []
        }
    else:
        info = {
            "status": "OK",
            "total_produtos": len(df),
            "colunas": df.columns.tolist(),
            "primeiros_5": df.head().to_dict('records'),
            "categorias": df['categoria'].unique().tolist(),
            "marcas": df['marca'].unique().tolist()
        }
    return jsonify(info)

# ---------------------------------------------
# ROTA API DE CADASTRO ATUALIZADA
# ---------------------------------------------
@app.route("/api/cadastro", methods=["POST"])
def api_cadastro():
    data = request.get_json()

    nome = data.get("nome")
    email = data.get("email")
    senha = data.get("senha")
    confirmar_senha = data.get("confirmar_senha")

    # Validações
    if not nome or not email or not senha:
        return jsonify({"erro": "Preencha todos os campos!"}), 400

    if senha != confirmar_senha:
        return jsonify({"erro": "As senhas não coincidem!"}), 400

    if len(senha) < 6:
        return jsonify({"erro": "A senha deve ter pelo menos 6 caracteres!"}), 400

    usuarios = carregar_usuarios()

    if email in usuarios:
        return jsonify({"erro": "Este email já está cadastrado!"}), 400

    # Salva usuário (em produção, hash a senha!)
    usuarios[email] = {
        "nome": nome,
        "senha": senha,  # EM PRODUÇÃO: usar bcrypt para hash!
        "data_cadastro": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
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

        # Verifica se é um usuário antigo (apenas senha) ou novo (dicionário)
        usuario = usuarios[email]
        if isinstance(usuario, dict):
            # Usuário novo (com nome)
            if usuario["senha"] != senha:
                return render_template("login.html", erro="Senha incorreta!")
        else:
            # Usuário antigo (apenas senha)
            if usuario != senha:
                return render_template("login.html", erro="Senha incorreta!")

        session["user"] = email
        return redirect("/catalogo")

    return render_template("login.html")

# -----------------------
# CADASTRO NORMAL
# -----------------------
@app.route("/cadastro", methods=["GET", "POST"])
def cadastro():
    if request.method == "POST":
        email = request.form["email"]
        senha = request.form["senha"]
        nome = request.form.get("nome_completo", "")

        usuarios = carregar_usuarios()

        if email in usuarios:
            return render_template("cadastro.html", erro="Este email já está cadastrado!")

        # Salva como dicionário com nome
        usuarios[email] = {
            "nome": nome,
            "senha": senha,
            "data_cadastro": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
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
        print(f"ERRO na página do produto: {e}")
        return "Erro interno", 500

# -----------------------
# APIs AUXILIARES - CORRIGIDAS
# -----------------------
@app.route("/api/maquiagens")
def api_maquiagens():
    try:
        if df.empty:
            return jsonify({"erro": "Nenhum produto disponível"}), 500
            
        produtos = df.to_dict('records')
        
        # Garante que todos os campos estão presentes e formatados corretamente
        for produto in produtos:
            produto['id'] = int(produto.get('id', 0))
            produto['preco'] = float(produto.get('preco', 0))
            produto['nome'] = produto.get('nome', 'Produto sem nome')
            produto['categoria'] = produto.get('categoria', 'Sem categoria')
            produto['marca'] = produto.get('marca', 'Sem marca')
            produto['tom_pele'] = produto.get('tom_pele', 'todos')
            produto['descricao'] = produto.get('descricao', 'Descrição não disponível')
            produto['imagem'] = produto.get('imagem', 'https://via.placeholder.com/300x300?text=Imagem+Não+Disponível')
        
        print(f"API: Retornando {len(produtos)} produtos")
        return jsonify(produtos)
    
    except Exception as e:
        print(f"ERRO na API /api/maquiagens: {e}")
        return jsonify({"erro": "Erro ao carregar produtos"}), 500

@app.route("/api/recomendar/<int:pid>")
def api_recomendar(pid):
    try:
        recomendacoes = recomendar(pid)
        return jsonify(recomendacoes)
    except Exception as e:
        print(f"ERRO na recomendação API: {e}")
        return jsonify([])

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