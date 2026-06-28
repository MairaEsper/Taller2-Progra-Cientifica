from estructura.biblia import Biblia
from estructura.visualizador import Visualizador

def main():
    print("Iniciando carga de datos")

    biblia = Biblia()

    ruta_dataset = '../data/t_asv.csv'
    ruta_keys = '../data/key_english.csv'
    
    try:
        biblia.cargar_datos(ruta_dataset, ruta_keys)
        print("Datos cargados correctamente\n")

        visualizador = Visualizador(biblia)
        visualizador.obtener_versiculos_por_libro()
        visualizador.obtener_distribucion_longitud_versiculos()
        visualizador.obtener_heatmap_similitud_libros()
    except Exception as e:
        print(f"Ocurrió un error al cargar los datos: {e}")
        return

if __name__ == "__main__":
    main()