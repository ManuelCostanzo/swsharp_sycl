# wget "https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete/uniprot_trembl.fasta.gz" -O uniprot_trembl.fasta.gz
# gunzip -k uniprot_trembl.fasta.gz

fileid="1k1oRjhZIZA7HzerQUaUqstYw0k-2JC-h"
filename="env_nr.fasta.tar.gz"

# Obtener el código de confirmación
confirm=$(curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}" | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')

# Descargar el archivo
if [ -z "$confirm" ]; then
    echo "No se pudo obtener el código de confirmación. Descargando sin confirmación..."
    curl -L -b ./cookie -o "$filename" "https://drive.google.com/uc?export=download&id=${fileid}"
else
    curl -L -b ./cookie -o "$filename" "https://drive.google.com/uc?export=download&id=${fileid}&confirm=${confirm}"
fi

# Limpia el archivo de cookies creado
rm ./cookie
tar -xzvf env_nr.fasta.tar.gz


