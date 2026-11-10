import hashlib
import json
import urllib.request

from src.config import ARQUIVOS, DADOS_DIR, ZENODO_RECORD

URL_API = f"https://zenodo.org/api/records/{ZENODO_RECORD}"


def md5(caminho):
    h = hashlib.md5()
    with open(caminho, "rb") as f:
        for bloco in iter(lambda: f.read(1 << 20), b""):
            h.update(bloco)
    return h.hexdigest()


def baixar():
    DADOS_DIR.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(URL_API) as resposta:
        registro = json.load(resposta)

    print(f"Dataset: {registro['metadata']['title']}")
    print(f"DOI: {registro['doi']} | Licença: {registro['metadata']['license']['id']}")

    for arquivo in registro["files"]:
        nome = arquivo["key"]
        if nome not in ARQUIVOS.values():
            continue
        destino = DADOS_DIR / nome
        checksum_esperado = arquivo["checksum"].split(":", 1)[1]
        if destino.exists() and md5(destino) == checksum_esperado:
            print(f"{nome}: já baixado e íntegro")
            continue
        print(f"Baixando {nome} ({arquivo['size'] / 1e6:.1f} MB)...")
        urllib.request.urlretrieve(arquivo["links"]["self"], destino)
        if md5(destino) != checksum_esperado:
            raise RuntimeError(f"Checksum MD5 divergente para {nome}")
        print(f"{nome}: ok (md5 {checksum_esperado})")


if __name__ == "__main__":
    baixar()
