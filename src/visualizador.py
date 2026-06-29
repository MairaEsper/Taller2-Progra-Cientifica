import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
from sklearn.decomposition import PCA
from tfidf import TfIdf

class Visualizador:
    def __init__(self, biblia):
        self.biblia = biblia
        self.tfidf = TfIdf()
    
    def obtener_versiculos_por_libro(self):
        datos = []

        for libro in self.biblia.testamentos["OT"].libros.values():
            total_versiculos = 0
            for capitulo in libro.capitulos.values():
                total_versiculos += len(capitulo.versiculos)
            
            datos.append((libro.nombre, total_versiculos))

        for libro in self.biblia.testamentos["NT"].libros.values():
            total_versiculos = 0
            for capitulo in libro.capitulos.values():
                total_versiculos += len(capitulo.versiculos)
            
            datos.append((libro.nombre, total_versiculos))

        df = pd.DataFrame(datos, columns=["Libro", "Cantidad"])

        plt.figure(figsize=(15, 6))
        plt.bar(df["Libro"], df["Cantidad"], color='blue')
        
        plt.xticks(rotation=90, fontsize=8) 
        
        plt.title("Cantidad Versículos por Libro")
        plt.xlabel("Libros")
        plt.ylabel("Número de Versículos")
        
        plt.show()

    def obtener_distribucion_longitud_versiculos(self):
        longitudes = []

        for libro in self.biblia.testamentos["OT"].libros.values():
            for capitulo in libro.capitulos.values():
                for versiculo in capitulo.versiculos:
                    cant_palabras = len(versiculo.texto_original.split())
                    longitudes.append(cant_palabras)

        for libro in self.biblia.testamentos["NT"].libros.values():
            for capitulo in libro.capitulos.values():
                for versiculo in capitulo.versiculos:
                    cant_palabras = len(versiculo.texto_original.split())
                    longitudes.append(cant_palabras)

        plt.figure(figsize=(10, 6))

        plt.hist(longitudes, bins=50, color='green', edgecolor='black')

        plt.title("Distribución de Longitud de Versículos")
        plt.xlabel("Cantidad de Palabras por Versículo")
        plt.ylabel("Cantidad de Versículos")
        
        plt.tight_layout()
        plt.show()

    def obtener_heatmap_similitud_libros(self):        
        libros = list(self.biblia.testamentos["OT"].libros.values()) + list(self.biblia.testamentos["NT"].libros.values())
        
        textos_por_libro = []
        for libro in libros:
            palabras_libro = []
            for capitulo in libro.capitulos.values():
                for versiculo in capitulo.versiculos:
                    texto_limpio = versiculo.texto_original.lower().replace(".", "").replace(",", "").replace('"', "")
                    palabras_libro.extend(texto_limpio.split())
            
            textos_por_libro.append(palabras_libro)
            
        vectores_tfidf = self.tfidf.calcular_tfidf(textos_por_libro)
            
        n = len(libros)
        matriz_similitud = np.zeros((n, n))
        nombres_libros = [libro.nombre for libro in libros]
        
        for i in range(n):
            for j in range(n):
                matriz_similitud[i][j] = self.tfidf.similitud_coseno(vectores_tfidf[i], vectores_tfidf[j])
                
        fig, ax = plt.subplots(figsize=(14, 12))
        
        cax = ax.imshow(matriz_similitud, cmap='viridis', interpolation='nearest')
        fig.colorbar(cax, label='Similitud del Coseno')
        
        ax.set_xticks(np.arange(n))
        ax.set_yticks(np.arange(n))
        ax.set_xticklabels(nombres_libros, rotation=90, fontsize=7)
        ax.set_yticklabels(nombres_libros, fontsize=7)
        
        plt.title("Heatmap de Similitud Semántica entre Libros")
        plt.tight_layout()
        plt.show()
        
    def obtener_pca_versiculos(self):
        textos_versiculos = []
        etiquetas_testamento = []

        for nombre_testamento, testamento in self.biblia.testamentos.items():
            for libro in testamento.libros.values():
                for capitulo in libro.capitulos.values():
                    for versiculo in capitulo.versiculos:
                        if len(versiculo.tokens) > 0:
                            palabras = versiculo.tokens
                        else:
                            texto_limpio = versiculo.texto_original.lower().replace(".", "").replace(",", "").replace('"', "")
                            palabras = texto_limpio.split()
                        if len(palabras) > 0: 
                            textos_versiculos.append(palabras)
                            etiquetas_testamento.append(nombre_testamento)

        vectores_tfidf = self.tfidf.calcular_tfidf(textos_versiculos)

        pca = PCA(n_components=2)
        matriz_tfidf = np.array(vectores_tfidf)
        coordenadas_2d = pca.fit_transform(matriz_tfidf)

        x_ot = []
        y_ot = []
        x_nt = []
        y_nt = []

        for i in range(len(etiquetas_testamento)):
            etiqueta = etiquetas_testamento[i]
    
            if etiqueta == "OT":
                x_ot.append(coordenadas_2d[i, 0])
                y_ot.append(coordenadas_2d[i, 1])
            else:
                x_nt.append(coordenadas_2d[i, 0])
                y_nt.append(coordenadas_2d[i, 1])
                
        plt.figure(figsize=(10, 8))

        plt.scatter(x_ot, y_ot, alpha=0.3, label='Antiguo Testamento (OT)', color='blue', s=10)
        plt.scatter(x_nt, y_nt, alpha=0.3, label='Nuevo Testamento (NT)', color='orange', s=10)

        plt.title("Visualización de Versículos utilizando PCA")
        plt.xlabel("Componente Principal 1")
        plt.ylabel("Componente Principal 2")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.show()