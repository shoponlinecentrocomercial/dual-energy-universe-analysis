#!/bin/bash
# Script para descargar los datos de Fermi-LAT 4FGL (14 años)

# Crear carpeta para los datos
mkdir -p ../data/fermi-4fgl
cd ../data/fermi-4fgl

# Base URL de tus archivos (reemplaza con la tuya)
BASE_URL="https://fermi.gsfc.nasa.gov/FTP/fermi/data/lat/queries/"

# Lista de archivos PH (reemplaza con los nombres de tus archivos)
FILES=(
    "L260721105812DDE961A079_PH00.fits"
    "L260721105812DDE961A079_PH01.fits"
    "L260721105812DDE961A079_PH02.fits"
    "L260721105812DDE961A079_PH03.fits"
    "L260721105812DDE961A079_PH04.fits"
    "L260721105812DDE961A079_PH05.fits"
    "L260721105812DDE961A079_PH06.fits"
    "L260721105812DDE961A079_SC00.fits"
)

# Descargar cada archivo
for FILE in "${FILES[@]}"; do
    echo "Descargando $FILE..."
    wget -q --show-progress "${BASE_URL}${FILE}"
done

echo "✅ Todos los archivos descargados en $(pwd)"