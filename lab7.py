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
path_traficogt = r"C:\Users\manue\OneDrive\Escritorio\Data_Science\Lab7\traficogt.txt"
path_tioberny = r"C:\Users\manue\OneDrive\Escritorio\Data_Science\Lab7\tioberny.txt"

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
        if 'id_str' in df.columns:
            df['tweet_text'] = df.apply(lambda row: json.dumps(row), axis=1)

    # Asegurar que el texto de los tweets sea string
    if 'tweet_text' in df.columns:
        df['tweet_text'] = df['tweet_text'].astype(str)
    else:
        print("Advertencia: 'tweet_text' no existe en el DataFrame.")
        return df

    # Aplicar la limpieza a los textos de los tweets
    df['cleaned_text'] = df['tweet_text'].apply(clean_tweet)
    
    # Verificar si todos los tweets están vacíos después de la limpieza
    if df['cleaned_text'].isnull().all():
        print("Advertencia: Todos los tweets están vacíos después de la limpieza.")
    
    # Eliminar duplicados
    df = df.drop_duplicates(subset=['tweet_text'])
    
    # Filtrar tweets que aún tengan contenido
    df = df[df['cleaned_text'].notnull()]
    
    # Mostrar algunas filas después de la limpieza
    debug_data(df, "tweets después de la limpieza")
    
    return df

# Función para construir el grafo
def build_graph(df):
    G = nx.Graph()  # Crear un grafo no dirigido
    if 'mentions' in df.columns:
        for _, row in df.iterrows():
            user = row['user'] if isinstance(row['user'], str) else row['user']['username']
            mentions = row['mentions']
            
            if pd.notnull(mentions):
                mentions = mentions if isinstance(mentions, list) else []
                for mention in mentions:
                    if mention:
                        G.add_edge(user, mention)
    return G

# Cargar datos
traficogt_df = load_json_data(path_traficogt)
tioberny_df = load_json_data(path_tioberny)

# Depuración inicial
debug_data(traficogt_df, "traficogt")
debug_data(tioberny_df, "tioberny")

# Preprocesar los datos
traficogt_df = preprocess_data(traficogt_df)
tioberny_df = preprocess_data(tioberny_df)

# Combinar ambos datasets
if traficogt_df is not None and tioberny_df is not None:
    all_tweets_df = pd.concat([traficogt_df, tioberny_df])
else:
    print("Error al cargar o procesar los datos.")

# Verificar si hay tweets vacíos
if all_tweets_df is not None and all_tweets_df['cleaned_text'].str.strip().eq('').all():
    print("Error: No hay texto suficiente para generar una nube de palabras.")
else:
    # Generar nube de palabras para ambos datasets por separado
    if not traficogt_df['cleaned_text'].isnull().all():
        wordcloud_traficogt = WordCloud(width=800, height=400).generate(' '.join(traficogt_df['cleaned_text'].dropna()))
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud_traficogt, interpolation='bilinear')
        plt.title("Nube de palabras - Traficogt")
        plt.axis('off')
        plt.show()
    
    if not tioberny_df['cleaned_text'].isnull().all():
        wordcloud_tioberny = WordCloud(width=800, height=400).generate(' '.join(tioberny_df['cleaned_text'].dropna()))
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud_tioberny, interpolation='bilinear')
        plt.title("Nube de palabras - Tioberny")
        plt.axis('off')
        plt.show()

# Construir el grafo no dirigido
G = build_graph(all_tweets_df)

# Detección de comunidades
partition = community_louvain.best_partition(G)

# Dibujar el grafo con las comunidades detectadas
pos = nx.spring_layout(G)
plt.figure(figsize=(10, 10))
nx.draw_networkx_nodes(G, pos, node_color=list(partition.values()), cmap=plt.cm.jet, node_size=500)
nx.draw_networkx_edges(G, pos, alpha=0.5)
nx.draw_networkx_labels(G, pos, font_size=12)
plt.show()

# Calcular centralidad de grado y mostrar los 10 usuarios más influyentes
degree_centrality = nx.degree_centrality(G)
top_10_influential_users = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:10]
print("Top 10 usuarios más influyentes por centralidad de grado:")
for user, centrality in top_10_influential_users:
    print(f"Usuario: {user}, Centralidad: {centrality}")


