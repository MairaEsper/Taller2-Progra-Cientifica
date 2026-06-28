from estructura.biblia import Biblia

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
            
    except Exception as e:
        print(f"Ocurrió un error al cargar los datos: {e}")
        return

if __name__ == "__main__":
    main()