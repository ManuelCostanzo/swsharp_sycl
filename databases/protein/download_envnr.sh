# mkdir env_nr
# cd env_nr
# # Descargar los archivos
# wget https://ftp.ncbi.nlm.nih.gov/blast/db/env_nr.00.tar.gz
# wget https://ftp.ncbi.nlm.nih.gov/blast/db/env_nr.01.tar.gz
# wget https://ftp.ncbi.nlm.nih.gov/blast/db/env_nr.02.tar.gz

# tar -zxvf env_nr.00.tar.gz
# tar -zxvf env_nr.01.tar.gz
# tar -zxvf env_nr.02.tar.gz


curl 'https://doc-14-9k-docs.googleusercontent.com/docs/securesc/h4tpl49irjrpbcmqk7o9l8d6roa10tlh/eqrefjsa2k0ont1qlqnacje4sjj4fg5q/1681752975000/03513909492646616532/03513909492646616532/1a3Qq5ec_oFoqmr8hvC-aHZDe8VVtUuxy?e=download&ax=ALy03A644cUWO_MCsC154VkGOan2NIsNh0bCLlrYh8KjxkpkCjh3Jz7mPQE_gYU9WYxw-23B6nMCzn0FiwqUmXc5uPJg8NBRJJ8dC6twSiECyD1z8kwKE4jyvM2WJ6NCmqdZ5KieeykP3wulDCYZ2yH_5TRiwWq1XbvsNmjDF7gUpNS4uKDbtQybC8OJy9VDcAd6uTmtYmtAmIIAQfGYuNpQoGu09xlAb1qxBpgoYqriJLoqNeP3n9q3y097SC6y9b80qMJ4T0Xg31qeKn71K_MPrmxOIa_4U_NPeaVEWcrFIZ6qN5wNShLUZlJ2dD6hvuhEQn4o0ac-u_IQAxAbkGBzYaKxvJsoWIg_13Hrg1LAeoYnmM5hmitusvonowTHxEtzt8IXB5QwbXXSTEM-1GENX1X1GtudmBufy_op1fdcMYt8w6gg77PuZXQfZXS2eOtT2WPYgYXS4xb2mXqx3ybHh36xjx5M-jLchIkqaXOz9vx3nCFHpHpm8-t9zPssz4M9KANZYX-M9pglY_mc9THDTTZvjJdr5smmWkU1Atxxx5dTChMAaEJUhGzb9vXESZ5KpEV5o0_7_kcdHm5PJ6YKaViEK32r5VkU_5VB6taDxHoKJ2ikE_Xj4IWzfEBGW4Q8-aTlYLUfWitaUx-gJMpukN57h5kbOb0W91xsDzj7AfqO9GV8i0MgbxraUJSoHL7GfDw3rHnWKS3S7zCBq7_uyl-NJ-gVLoarnRqUDpMUXlWL0jvCjAixY7fw61hQ80tCmemRbsK9iEPwxf3XvEbgBYy1z0GSTqEtRiyd1pTuvyrkLeMwblb1jJngGR1SJCHtL2RtQzJECCg0LolWE3Jnc45HY87TgEuBot-lmWiE49gjQ0YCsbCG8AxHC89Jb04&uuid=52e3bea6-5f35-425d-8b2e-7dd228d8d8a0&authuser=0&nonce=e8ahp0748qqe4&user=03513909492646616532&hash=gr1tbbsfu3i63db8p9ffo1ffr8f6res8' \
-H 'authority: doc-14-9k-docs.googleusercontent.com' \
-H 'accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7' \
-H 'accept-language: en-US,en;q=0.9,es-US;q=0.8,es;q=0.7' \
-H 'cookie: AUTH_83n3loam9t952vpaqik1o7d1t6uerd2b_nonce=e8ahp0748qqe4' \
-H 'sec-ch-ua: "Chromium";v="112", "Google Chrome";v="112", "Not:A-Brand";v="99"' \
-H 'sec-ch-ua-mobile: ?0' \
-H 'sec-ch-ua-platform: "Linux"' \
-H 'sec-fetch-dest: iframe' \
-H 'sec-fetch-mode: navigate' \
-H 'sec-fetch-site: cross-site' \
-H 'upgrade-insecure-requests: 1' \
-H 'user-agent: Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/112.0.0.0 Safari/537.36' \
-H 'x-client-data: CIa2yQEIo7bJAQipncoBCOjpygEIkqHLAQjhl80BCOSXzQEI7Z7NAQiHoM0BCImhzQEIvqLNAQifpM0BCJSlzQEIrIStAg==' \
--compressed --output envnr.fasta.tar.gz

tar -xzvf envnr.fasta.tar.gz