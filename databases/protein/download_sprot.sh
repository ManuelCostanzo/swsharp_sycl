#wget "https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete/uniprot_sprot.fasta.gz" -O uniprot_sprot.fasta.gz
#gunzip -k uniprot_sprot.fasta.gz

curl 'https://doc-04-9k-docs.googleusercontent.com/docs/securesc/h4tpl49irjrpbcmqk7o9l8d6roa10tlh/8oo1905ifumgbumep3aj83tgdavaqeq0/1681752600000/03513909492646616532/03513909492646616532/1JorMVhY_jHZmok>
  -H 'authority: doc-04-9k-docs.googleusercontent.com' \
  -H 'accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7' \
  -H 'accept-language: en-US,en;q=0.9,es-US;q=0.8,es;q=0.7' \
  -H 'cookie: AUTH_83n3loam9t952vpaqik1o7d1t6uerd2b_nonce=levklhvi6j4do' \
  -H 'sec-ch-ua: "Chromium";v="112", "Google Chrome";v="112", "Not:A-Brand";v="99"' \
  -H 'sec-ch-ua-mobile: ?0' \
  -H 'sec-ch-ua-platform: "Linux"' \
  -H 'sec-fetch-dest: iframe' \
  -H 'sec-fetch-mode: navigate' \
  -H 'sec-fetch-site: cross-site' \
  -H 'upgrade-insecure-requests: 1' \
  -H 'user-agent: Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/112.0.0.0 Safari/537.36' \
  -H 'x-client-data: CIa2yQEIo7bJAQipncoBCOjpygEIkqHLAQjhl80BCOSXzQEI7Z7NAQiHoM0BCImhzQEIvqLNAQifpM0BCJSlzQEIrIStAg==' \
  --compressed --output uniport_sprot.fasta.tar.gz

tar -xzvf uniport_sprot.fasta.tar.gz
