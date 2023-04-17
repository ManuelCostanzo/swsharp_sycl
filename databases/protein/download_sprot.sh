#wget "https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete/uniprot_sprot.fasta.gz" -O uniprot_sprot.fasta.gz
#gunzip -k uniprot_sprot.fasta.gz

curl 'https://doc-04-9k-docs.googleusercontent.com/docs/securesc/h4tpl49irjrpbcmqk7o9l8d6roa10tlh/8oo1905ifumgbumep3aj83tgdavaqeq0/1681752600000/03513909492646616532/03513909492646616532/1JorMVhY_jHZmok3Gktjkg9y5CYT7jemQ?e=download&ax=ALy03A4FwALQZQVDowb0wUg-qiq82YX90FaoiSLGgSNo8oV8yAdh-Loh2Sj39tjwRZkOrd-UiyrAnt1UJ_EM2li39YAivNQO6DH4DyUYqNfodrF-IVVRlhd6F65urgDfZAhvgxD-LR8EjH48sbDhU5xSVpws75oBMeNVksAz4UjZOt1lfv0YH6Q9kwcx79xPH5W_dUSR40oPmVkGR5ZI24uwvQvnLdBzqdDQ01b4A8rT8NVcslcz7GXawvTPJpax1CEDF2w6bigv5tJV0_k2U2gc0MGvWGLS93X8x5MWaa40bmDr2XmPcKQVlj7YGkNAfBgJZNqgVsjCK58F5FRnMKKjDDqTPVG6rMzfQGF3qcbMe1gbmL_poOJpby3hDPqAv_Y8Mxp53HmCu87TkU5ESmdImQAZ5sigfGGW3LoJryvSZv6q8LJEA37R2mezegpgt8rEj69Gux31dSvFc8eWHQIG4Et7zxZf70KVQcQ7DiJmMNrkZMCzITerSx4XQcP7wqIwA9761-ZtJQA4-v0gPLr96_c2wH6l3tn066LSqshdYxGqBApfmQMKhKqdQs_O1MzyV2D4AdfTGGm26xONYqvhQzuOCH42isWHBXPgpHPzuv6vmzsEIXHhLpW1-8GHmQEi-hptCDVRRRa0jNA7tQR4pLER35gD1u2sK3RrcJwhdwkN_8rangugKd5APZeG4i1uilbVs6pF3K_V1Gc4aW66rpxwp4BQ9qGC4Ld1xq5l1I_SZby3HnIKcDZVB7YMaJA56L5JO9ApCWn30O9V6hQFJozYQyh84HXEt-CS-gR_gDtEjY3_lGw7MH1Bzm0zBkY7L07msaH8I0xhCkKp4R29veU1xfi7Lr8b8dsq39YO6Bn62qlW1aMrDgQj-DvFbBY&uuid=f13445be-f765-4db9-8905-a2c1c3259431&authuser=0&nonce=levklhvi6j4do&user=03513909492646616532&hash=6s30gd2gumu5gktnlvm9a38eusnqvrel' \
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
--compressed --output uniport.fasta.tar.gz

tar -xzvf uniport.fasta.tar.gz