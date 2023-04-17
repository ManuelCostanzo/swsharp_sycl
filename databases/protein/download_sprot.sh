#wget "https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete/uniprot_sprot.fasta.gz" -O uniprot_sprot.fasta.gz
#gunzip -k uniprot_sprot.fasta.gz

wget 'https://doc-04-9k-docs.googleusercontent.com/docs/securesc/h4tpl49irjrpbcmqk7o9l8d6roa10tlh/a74vsfupjqiig5m0qqfiag9t1j27650i/1681757550000/03513909492646616532/03513909492646616532/1JorMVhY_jHZmok3Gktjkg9y5CYT7jemQ?e=download&ax=ALy03A6s7whNXi4D5bxTQE7LTLhXckWeJq9zzYR7haA5-Vvtctjq0bD6D-_KNjTM9XX_DfukbD99sX-JzAtsR2k9ptP0YaSKhOzt1-oaMAL6IvJFEzwlGXT5HX3la-Ic5fjm7EZmcl6715gUVSVCPBfDOmPn3BVvhTaJ4dacwQBYA7gsdjHeH2lRLHLdMbVDoiB45tJHJBSZZsNgOhdWRYtEXlTRo-trFr9yCKohP0IXz6mYNXnXJZB0SleqdHu_BAPi3RFyv8opxsXq446u8BdRaX6dzYROay6gthrJ9tpDFaWjIRQYCuU-JIPpb2eNOETYWnBrRur3dJdP5BxmxInz5SmFzE0RUBCZj2XjT-woWjnq-6b1R_U5r5j0r8fOvrTj3spdUZa5alZ5rFRcOD8RhdR7upTbtOhosB_iq_j18jzYYA4O_rxYRKxHOkG2S6rY30vdtN23a6YqXEbiFxLQWjOnG-Vwro_QovfEhPRuErlgm2bbc85ZJIdKfvtCdwG9_jpnyWb9YhS-nqhEkOKWK7SgTBzMNV_YXF7mEuDyNCsBKBX9t5Rg54KMnCrsXDpcs56teHaQjWKZrEPxkvlvAju8JcTeMnV5sWSEcSlvNAH0OL3TcYFfajNmIwFPJ1CcPkA_WEoVSXffSkiLhDQK98jooJK5CQLVDUMqPK453awhjAB180yYmRZbKgZ5Zi3hGUv2M766ijhuG9t-4S1sg2dIGd_QnhpISIQ_OmBLMdSPAMcuI6hPEqcv5Dh3m_Vjc4XdVFg5nF3Bq8kG9aW34XkU_MDJUGO4Lt5BQtfwkXgV_H6wdzMOID7h4NnvozdYxnXWQjKAyzst-aMgKJKCAjmtsq9_JulTOPbfzfE3Pr9W6oSI_Ms3yuX_0E7Mvhs&uuid=ab25e724-ade8-4524-9319-49d392bb6fec&authuser=0' \
--header 'authority: doc-04-9k-docs.googleusercontent.com' \
--header 'accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7' \
--header 'accept-language: en-US,en;q=0.9,es-US;q=0.8,es;q=0.7' \
--header 'cookie: AUTH_83n3loam9t952vpaqik1o7d1t6uerd2b=03513909492646616532|1681757475000|9pscr43gp8u54qrlk3oukihcvlrig3f6' \
--header 'sec-ch-ua: "Chromium";v="112", "Google Chrome";v="112", "Not:A-Brand";v="99"' \
--header 'sec-ch-ua-mobile: ?0' \
--header 'sec-ch-ua-platform: "Linux"' \
--header 'sec-fetch-dest: iframe' \
--header 'sec-fetch-mode: navigate' \
--header 'sec-fetch-site: cross-site' \
--header 'upgrade-insecure-requests: 1' \
--user-agent 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/112.0.0.0 Safari/537.36' \
--header 'x-client-data: CIa2yQEIo7bJAQipncoBCOjpygEIkqHLAQjhl80BCOSXzQEI7Z7NAQiHoM0BCImhzQEIvqLNAQifpM0BCJSlzQEIrIStAg==' \
--compression=auto \
--output-document=uniprot_sprot.fasta.tar.gz

tar -xzvf uniprot_sprot.fasta.tar.gz
