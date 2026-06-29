from estructura.biblia import Biblia

from visualizador import Visualizador
from buscador import Buscador
from clasificador import ClasificadorVersiculos
from generador import Generador
from sentimientos import AnalizadorSentimiento

def menu_principal():
    print("----- Menú -----")
    print("1. Preprocesamiento de texto")
    print("2. Visualización y análisis exploratorio")
    print("3. Visualización de versículos utilizando PCA")
    print("4. Motor de búsqueda semántico")
    print("5. Clasificador de versículos")
    print("6. Modelo generador de texto")
    print("7. Análisis de sentimiento a lo largo de los capítulos")
    print("0. Salir")
    return input("Seleccione una opción (0-7): ")

def main():    
    biblia = Biblia()
    ruta_dataset = '../data/t_asv.csv'
    ruta_keys = '../data/key_english.csv'
    
    try:
        biblia.cargar_datos(ruta_dataset, ruta_keys)
    except Exception as e:
        print(f"Ocurrió un error al cargar los datos: {e}")

    while True:
        opcion = menu_principal()
        if opcion == '1':
            print(f"Palabras únicas: {len(biblia.vocabulario)}")
            top = sorted(biblia.frecuencias_globales.items(), key=lambda x: x[1], reverse=True)[:5]
            print("Las 5 palabras más frecuentes:")
            for palabra, frec in top:
                print(f" - {palabra}: {frec}")

        elif opcion == '2':
            visualizador = Visualizador(biblia)
            visualizador.obtener_versiculos_por_libro()
            visualizador.obtener_distribucion_longitud_versiculos()
            visualizador.obtener_heatmap_similitud_libros()

        elif opcion == '3':
            visualizador = Visualizador(biblia)
            visualizador.obtener_pca_versiculos()

        elif opcion == '4':
            buscador = Buscador(biblia)
            frase_buscar = input("Ingrese la frase que desea buscar: ")
            buscador.procesar_biblia()
            buscador.buscar_frase(frase_buscar, 5)

        elif opcion == '5':
            clasificador = ClasificadorVersiculos(biblia)
            clasificador.entrenar_evaluar()

        elif opcion == '6':
            generador = Generador(biblia)
            generador.entrenar_modelos(n_maximo=4)
            palabra = input("Ingresa una palabra inicial (ej. 'the', 'god', 'jesus'): ")
            try:
                n = int(input("Ingresa el valor de N para el modelo (1-4): "))
                if 1 <= n <= 4:
                    versiculo_generado = generador.generar_versiculo(n, palabra)
                    print(f"Texto generado (N={n}):")
                    print(f"   {versiculo_generado.capitalize()}")
                else:
                    print("Error: El valor de N debe estar entre 1 y 4.")
            except ValueError:
                print("Error: Debes ingresar un número entero.")

        elif opcion == '7':
            analizador = AnalizadorSentimiento(biblia)
            analizador.calcular_visualizar()

        elif opcion == '0':
            print("Fin.")
            break

        else:
            print("Opción no válida. Intentar nuevamente.")

if __name__ == "__main__":
    main()