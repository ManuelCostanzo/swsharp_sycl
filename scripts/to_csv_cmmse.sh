#!/bin/bash

folder="CMMSE"
output_file="times.csv"

# Obtener una lista de todos los archivos .txt en la carpeta
files=($folder/*.txt)

# Crear un array para cada archivo
declare -A file_times

# Máximo número de tiempos encontrados en un archivo
max_count=0

# Procesar cada archivo
for file in "${files[@]}"; do
    filename=$(basename "$file" .txt)
    
    # Leer cada línea del archivo y extraer los tiempos
    i=0
    while read -r line; do
        if [[ "$line" =~ TIME:\ ([0-9]*\.[0-9]*) ]]; then
            file_times[$filename,$i]=${BASH_REMATCH[1]}
            ((i++))
        fi
    done < "$file"
    
    # Actualizar el conteo máximo si es necesario
    if ((i > max_count)); then
        max_count=$i
    fi
done

# Crear el archivo CSV
{
    # Encabezado con los nombres de archivo
    for file in "${files[@]}"; do
        filename=$(basename "$file" .txt)
        printf "%s," "$filename"
    done
    printf "\n"
    
    # Datos de tiempos
    for ((i=0; i<max_count; i++)); do
        for file in "${files[@]}"; do
            filename=$(basename "$file" .txt)
            printf "%s," "${file_times[$filename,$i]}"
        done
        printf "\n"
    done
} > "$output_file"

echo "Archivo CSV generado: $output_file"
