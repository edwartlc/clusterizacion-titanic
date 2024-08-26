from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import altair as alt
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

titanic = sns.load_dataset('titanic')

st.title('Clusterización del conjunto de datos Titanic con KMeans')
st.write(titanic.head())

df = titanic[['age', 'fare', 'pclass', 'survived', 'sibsp']].dropna()
st.write(df.head())

scaler = StandardScaler()
dfTransformed = scaler.fit_transform(df)
st.write(dfTransformed)

nClusters = st.slider('Seleccione el número de clusters', 2, 10, 3)

kmeans = KMeans(n_clusters = nClusters, random_state=42)
cluster = kmeans.fit_predict(dfTransformed)
df['Cluster'] = cluster
st.write(df)

chart = alt.Chart(df).mark_circle(size=60).encode(
    x = 'age',
    y = 'fare',
    color = 'Cluster:N'
).interactive()

st.altair_chart(chart)

# Aplicar PCA (Principal Component Analisis) para reducir a 2 dimensiones

pca = PCA(n_components = 2)

dfPca = pca.fit_transform(dfTransformed)

dfPca = pd.DataFrame(dfPca, columns=['PCA1', 'PCA2'])
dfPca['Cluster'] = cluster
st.write(dfPca)

st.title('Clusterización del conjunto de datos Titanic con KMeans (2D)')
chart = alt.Chart(dfPca).mark_circle(size=60).encode(
    x = 'PCA1',
    y = 'PCA2',
    color = 'Cluster:N'
).interactive()

st.altair_chart(chart)