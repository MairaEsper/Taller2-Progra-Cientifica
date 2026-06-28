from estructura.biblia import Biblia
from estructura.visualizador import Visualizador
from buscador import Buscador

def main():
    print("Iniciando carga de datos")
    biblia = Biblia()

    ruta_dataset = '../data/t_asv.csv'
    ruta_keys = '../data/key_english.csv'

    try:
        biblia.cargar_datos(ruta_dataset, ruta_keys)
        print("Datos cargados correctamente\n")
        
        print(f"Palabras únicas: {len(biblia.vocabulario)}")
        top = sorted(biblia.frecuencias_globales.items(), key=lambda x: x[1], reverse=True)[:5]
        print("Las 5 palabras más frecuentes:")
        for palabra, frec in top:
            print(f" - {palabra}: {frec}")
            

        visualizador = Visualizador(biblia)
        visualizador.obtener_versiculos_por_libro()
        visualizador.obtener_distribucion_longitud_versiculos()
        visualizador.obtener_heatmap_similitud_libros()


        buscador = Buscador(biblia)
        print("--- Buscador Semántico ---")
        frase_buscar = input("Ingrese la frase que desea buscar: ")
        buscador.procesar_biblia()
        buscador.buscar_frase(frase_buscar, 5)


        visualizador.obtener_pca_versiculos()
    except Exception as e:
        print(f"Ocurrió un error al cargar los datos: {e}")
        return

if __name__ == "__main__":
    main()