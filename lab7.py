import pandas as pd
import re
import json
import networkx as nx
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import community as community_louvain
from nltk.corpus import stopwords
import chardet

# Ruta de archivos
path_traficogt = r".\traficogt.txt"
path_tioberny = r".\tioberny.txt"

# Función para detectar la codificación del archivo
def detect_encoding(file_path):
    with open(file_path, 'rb') as f:
        result = chardet.detect(f.read())
        return result['encoding']

# Función para cargar datos en formato JSON, con manejo de errores
def load_json_data(file_path):
    encoding = detect_encoding(file_path)
    data = []
    with open(file_path, 'r', encoding=encoding) as file:
        for line in file:
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"Advertencia: Se omitió una línea con un error de decodificación en {file_path}")
                continue
    return pd.DataFrame(data)

# Función de depuración para revisar el contenido
def debug_data(df, name):
    if df is None or df.empty:
        print(f"Advertencia: {name} está vacío después de la carga.")
    else:
        print(f"Mostrando las primeras 5 filas de {name}:")
        print(df.head())

# Función para limpiar tweets
def clean_tweet(text):
    if not isinstance(text, str):
        return text
    # Convertir a minúsculas
    text = text.lower()
    # Quitar URLs
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
    # Eliminar solo caracteres no alfanuméricos (excepto menciones y hashtags)
    text = re.sub(r'[^\w\s@#]', '', text)
    return text if text.strip() else None

# Función para preprocesar los datos
def preprocess_data(df):
    if df is None:
        return None

    # Verificar que el campo 'rawContent' existe
    if 'rawContent' in df.columns:
        df['tweet_text'] = df['rawContent']
    else:
        # Si no existe 'rawContent', tratar de extraer el contenido de otra forma
        if 'text' in df.columns:
            df['tweet_text'] = df['text']
        else:
            print("No se encontró un campo de texto adecuado para los tweets")
            return None

    # Aplicar la limpieza de tweets
    df['tweet_text_clean'] = df['tweet_text'].apply(clean_tweet)

    # Quitar tweets duplicados
    df.drop_duplicates(subset='tweet_text_clean', inplace=True)

    return df

# Función para extraer menciones, retweets y respuestas
def extract_interactions(df):
    if df is None:
        return None

    # Extraer menciones de usuarios
    df['mentions'] = df['tweet_text_clean'].apply(lambda x: re.findall(r'@\w+', x) if isinstance(x, str) else [])

    # Extraer retweets
    df['is_retweet'] = df['tweet_text_clean'].apply(lambda x: True if isinstance(x, str) and x.startswith('rt @') else False)

    # Extraer respuestas (asumimos que las respuestas contienen "@" al inicio del texto)
    df['is_reply'] = df['tweet_text_clean'].apply(lambda x: True if isinstance(x, str) and x.startswith('@') else False)

    return df

# Función para construir la red de interacciones entre usuarios
def build_interaction_graph(df):
    if df is None:
        return None

    G = nx.DiGraph()

    for _, row in df.iterrows():
        user = row['user']['username'] if 'user' in row and isinstance(row['user'], dict) else None
        mentions = row['mentions']

        if user:
            for mention in mentions:
                G.add_edge(user, mention.replace('@', ''), type='mention')

            if row['is_retweet']:
                retweeted_user = mentions[0] if mentions else None
                if retweeted_user:
                    G.add_edge(user, retweeted_user.replace('@', ''), type='retweet')

            if row['is_reply']:
                replied_user = mentions[0] if mentions else None
                if replied_user:
                    G.add_edge(user, replied_user.replace('@', ''), type='reply')

    return G

# Función para análisis exploratorio
def exploratory_analysis(df):
    if df is None:
        return None

    # Número de tweets
    num_tweets = len(df)
    print(f"Número total de tweets: {num_tweets}")

    # Usuarios únicos
    unique_users = df['user'].apply(lambda x: x['username'] if 'user' in x else None).nunique()
    print(f"Número de usuarios únicos: {unique_users}")

    # Menciones más frecuentes
    mentions_series = df['mentions'].explode()
    top_mentions = mentions_series.value_counts().head(10)
    print("Menciones más frecuentes:")
    print(top_mentions)

    # Hashtags más frecuentes
    df['hashtags'] = df['tweet_text_clean'].apply(lambda x: re.findall(r'#\w+', x) if isinstance(x, str) else [])
    hashtags_series = df['hashtags'].explode()
    top_hashtags = hashtags_series.value_counts().head(10)
    print("Hashtags más frecuentes:")
    print(top_hashtags)

    # Generar una nube de palabras de hashtags
    wordcloud = WordCloud(width=800, height=400).generate(' '.join(hashtags_series.dropna()))
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()

# Carga y procesamiento de datos de @traficogt
df_traficogt = load_json_data(path_traficogt)
debug_data(df_traficogt, "TráficoGT")
df_traficogt = preprocess_data(df_traficogt)
df_traficogt = extract_interactions(df_traficogt)
exploratory_analysis(df_traficogt)

# Crear el grafo de interacciones para @traficogt
G_traficogt = build_interaction_graph(df_traficogt)

# Visualización del grafo de interacciones
plt.figure(figsize=(10, 10))
pos = nx.spring_layout(G_traficogt, k=0.1)
nx.draw(G_traficogt, pos, with_labels=True, node_size=50, node_color="skyblue", font_size=8)
plt.title("Red de interacciones de @traficogt")
plt.show()

# Cálculo de métricas de red
density = nx.density(G_traficogt)
diameter = nx.diameter(G_traficogt) if nx.is_connected(G_traficogt.to_undirected()) else "Red no conectada"
clustering_coefficient = nx.average_clustering(G_traficogt.to_undirected())

print(f"Densidad de la red: {density}")
print(f"Diámetro de la red: {diameter}")
print(f"Coeficiente de agrupamiento: {clustering_coefficient}")
