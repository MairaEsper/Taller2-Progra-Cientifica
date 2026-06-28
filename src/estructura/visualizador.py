import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math

class Visualizador:
    def __init__(self, biblia):
        self.biblia = biblia
    
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

    def similitud_coseno(self, vector_a, vector_b):
        producto_punto = np.dot(vector_a, vector_b)
        magnitud_a = np.sqrt(np.sum(vector_a**2))
        magnitud_b = np.sqrt(np.sum(vector_b**2))
        
        if magnitud_a == 0 or magnitud_b == 0:
            return 0.0
        return producto_punto / (magnitud_a * magnitud_b)
    

    def obtener_heatmap_similitud_libros(self):        
        libros = list(self.biblia.testamentos["OT"].libros.values()) + list(self.biblia.testamentos["NT"].libros.values())
        
        textos_por_libro = []
        frecuencias_documento = {}
        
        for libro in libros:
            palabras_libro = []
            for capitulo in libro.capitulos.values():
                for versiculo in capitulo.versiculos:
                    texto_limpio = versiculo.texto_original.lower().replace(".", "").replace(",", "").replace('"', "")
                    palabras_libro.extend(texto_limpio.split())
            
            textos_por_libro.append(palabras_libro)
            
            for palabra in set(palabras_libro):
                frecuencias_documento[palabra] = frecuencias_documento.get(palabra, 0) + 1
                
        vocabulario = list(frecuencias_documento.keys())
        total_libros = len(libros)
        
        idf_dict = {}
        for palabra, df in frecuencias_documento.items():
            idf_dict[palabra] = math.log10(total_libros / df)
            
        vectores_tfidf = []
        for palabras_libro in textos_por_libro:
            total_palabras = len(palabras_libro)

            conteo_tf = {}
            for palabra in palabras_libro:
                conteo_tf[palabra] = conteo_tf.get(palabra, 0) + 1
            
            vector_libro = np.zeros(len(vocabulario))
            for i, palabra in enumerate(vocabulario):
                if palabra in conteo_tf:
                    tf = conteo_tf[palabra] / total_palabras
                    vector_libro[i] = tf * idf_dict[palabra]
                    
            vectores_tfidf.append(vector_libro)
            
        n = len(libros)
        matriz_similitud = np.zeros((n, n))
        nombres_libros = [libro.nombre for libro in libros]
        
        for i in range(n):
            for j in range(n):
                matriz_similitud[i][j] = self.similitud_coseno(vectores_tfidf[i], vectores_tfidf[j])
                
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