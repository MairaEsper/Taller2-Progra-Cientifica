import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
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