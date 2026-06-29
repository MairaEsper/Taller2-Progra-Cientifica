import pandas as pd
import matplotlib.pyplot as plt
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

nltk.download('vader_lexicon')

class AnalizadorSentimiento:
    def __init__(self, biblia):
        self.biblia = biblia
        self.sia = SentimentIntensityAnalyzer()

    def calcular_visualizar(self):
        resultados_capitulos = []

        for testamento in self.biblia.testamentos.values():
            for libro in testamento.libros.values():
                for capitulo in libro.capitulos.values():
                    total_versiculos = len(capitulo.versiculos)
                    if total_versiculos == 0:
                        continue
                        
                    sentimiento_acumulado = 0
                    
                    for versiculo in capitulo.versiculos:
                        puntaje = self.sia.polarity_scores(versiculo.texto_original)['compound']
                        sentimiento_acumulado += puntaje

                    promedio_capitulo = sentimiento_acumulado / total_versiculos
                    
                    resultados_capitulos.append({
                        'Libro': libro.nombre,
                        'Capitulo': capitulo.numero,
                        'Sentimiento': promedio_capitulo
                    })
                    
        df_sentimientos = pd.DataFrame(resultados_capitulos)

        capitulos_ordenados = df_sentimientos.sort_values(by='Sentimiento')
        
        print("Los 5 capítulos con sentimiento más negativo:")
        print(capitulos_ordenados.iloc[:5].to_string(index=False))
        
        print("Los 5 capítulos con sentimiento más positivo:")
        print(capitulos_ordenados.iloc[-5:].to_string(index=False))

        promedio_por_libro = df_sentimientos.groupby('Libro', sort=False)['Sentimiento'].mean().reset_index()

        plt.figure(figsize=(15, 6))
        plt.plot(promedio_por_libro['Libro'], promedio_por_libro['Sentimiento'], marker='o', color='purple', linestyle='-')

        plt.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        
        plt.title("Evolución del Sentimiento a lo largo de los Libros de la Biblia", fontsize=14)
        plt.xlabel("Libro", fontsize=12)
        plt.ylabel("Sentimiento Promedio (Puntaje Compound)", fontsize=12)
        plt.xticks(rotation=90, fontsize=8)
        
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.tight_layout()
        plt.show()