import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from tfidf import TfIdf

class ClasificadorVersiculos:
    def __init__(self, biblia):
        self.biblia = biblia
        self.tfidf = TfIdf()

    def entrenar_evaluar(self):
        textos_versiculos = []
        etiquetas_libros = []

        for testamento in self.biblia.testamentos.values():
            for libro in testamento.libros.values():
                for capitulo in libro.capitulos.values():
                    for versiculo in capitulo.versiculos:
                        if len(versiculo.tokens) > 0:
                            textos_versiculos.append(versiculo.tokens)
                            etiquetas_libros.append(libro.nombre)

        vectores_tfidf = self.tfidf.calcular_tfidf(textos_versiculos)

        x = np.array(vectores_tfidf)
        y = np.array(etiquetas_libros)

        np.random.seed(42)

        limite = int(len(x) * 0.8)

        indices = np.random.permutation(len(x))
        x_barajado = x[indices]
        y_barajado = y[indices]

        x_train = x_barajado[:limite]
        x_test = x_barajado[limite:]
        
        y_train = y_barajado[:limite]
        y_test = y_barajado[limite:]

        modelo = LogisticRegression(max_iter=1000)
        modelo.fit(x_train, y_train)

        y_pred = modelo.predict(x_test)

        accuracy = accuracy_score(y_test, y_pred) * 100
        print(f"Exactitud del modelo (accuracy): {round(accuracy, 2)}%")

        cm = confusion_matrix(y_test, y_pred)

        fig, ax = plt.subplots(figsize=(20, 20))

        cax = ax.imshow(cm, cmap='Blues', interpolation='nearest')
        
        plt.title("Matriz de Confusión - Clasificación de Versículos por Libro", fontsize=16)

        nombres_libros = modelo.classes_
        n = len(nombres_libros)
        
        ax.set_xticks(np.arange(n))
        ax.set_yticks(np.arange(n))
        ax.set_xticklabels(nombres_libros, rotation=90, fontsize=8)
        ax.set_yticklabels(nombres_libros, fontsize=8)
        
        plt.tight_layout()
        plt.show()