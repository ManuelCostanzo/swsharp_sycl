#wget "https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete/uniprot_sprot.fasta.gz" -O uniprot_sprot.fasta.gz
#gunzip -k uniprot_sprot.fasta.gz

wget --no-check-certificate -O uniprot_sprot.fasta.tar.gz 'https://drive.google.com/u/0/uc?id=1JorMVhY_jHZmok3Gktjkg9y5CYT7jemQ&export=download&confirm=t&uuid=dc13607b-2d82-4b25-8a15-e048d5f16311&at=ANzk5s630tyH6ohY-hBaXlqE6CER:1681758399201'

tar -xzvf uniprot_sprot.fasta.tar.gz

