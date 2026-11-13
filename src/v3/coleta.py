"""Coleta da base v3: Shorts de canais faceless e animados (explicações, "e se", ciência, curiosidades), em várias línguas.

Etapas (cada uma retoma de onde parou):
  descobrir  buscas no YouTube (yt-dlp, sem cota) -> canais que postam Shorts no nicho      -> data/v3/canais_candidatos.json
  abas       aba Shorts completa de cada canal (id, título, views)                           -> data/v3/abas/<canal_id>.jsonl
  api        YouTube Data API (videos.list/channels.list, 1 unidade por 50 ids): data,
             duração, tags, língua, descrição, curtidas, comentários; inscritos do canal       -> data/v3/videos_api.jsonl, canais_api.json
  capas      capa vertical (oar2.jpg) de cada Short                                           -> data/v3/capas/<id>.jpg

Uso: .venv/bin/python -m src.v3.coleta descobrir|abas|api|capas|tudo
Só dados públicos. A API usa o token de leitura do canal (~/.config/yt-publicar/tokens/simulando.analise.json).
"""
import argparse
import json
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

from src.v3.config import ABAS, CAPAS, SNAPUGC, TOKENS, V3, YTDLP

# Buscas do nicho do Simulando o Mundo e vizinhos: hipóteses ("e se"), animações 3D explicativas, ciência, espaço,
# corpo humano, engenharia, história e curiosidades. Em várias línguas, porque os formatos que viralizam se repetem.
BUSCAS = {
    "pt": ["e se a terra parasse de girar", "e se o sol sumisse", "o que aconteceria se", "e se você caísse num buraco negro",
           "simulação e se", "animação 3d como funciona", "como funciona por dentro animação", "curiosidades animação 3d",
           "e se a lua caísse", "e se a gravidade sumisse", "o que acontece com seu corpo se", "e se os oceanos secassem",
           "explicação animada ciência", "fatos sobre o espaço animação", "como é feito animação", "e se humanos",
           "e se um asteroide caísse no brasil", "curiosidades do corpo humano animação", "história animada curiosidades",
           "engenharia animação como funciona", "e se você vivesse em marte", "e se a internet acabasse",
           "o que aconteceria com a terra se", "simulação física", "por que o céu é azul animação", "animação explicativa",
           "você sabia animação", "e se todos pulassem ao mesmo tempo", "e se não existisse", "e se dinossauros voltassem"],
    "en": ["what if the earth stopped spinning", "what if the sun disappeared", "what would happen if", "what if you fell into a black hole",
           "what if animation", "3d animation how it works", "how it works inside animation", "3d animation explained",
           "what happens to your body if", "what if the moon crashed", "what if gravity disappeared", "zack d films style",
           "satisfying 3d explanation", "how surgery works animation", "how engines work animation", "space facts animation",
           "what if humans", "what if dinosaurs", "history animated facts", "science animation shorts",
           "how is it made animation", "what if you lived on mars", "physics simulation what if", "engineering explained animation",
           "why does animation explained", "human body animation", "what if you never slept", "what if oceans dried up",
           "animated explainer", "did you know animation"],
    "es": ["qué pasaría si", "y si la tierra dejara de girar", "y si el sol desapareciera", "animación 3d cómo funciona",
           "cómo funciona por dentro animación", "curiosidades animación", "qué le pasa a tu cuerpo si", "y si la luna",
           "simulación qué pasaría", "datos del espacio animación", "explicación animada ciencia", "historia animada curiosidades"],
    "de": ["was wäre wenn", "was passiert wenn animation", "wie funktioniert animation 3d", "erklärt animation",
           "was wäre wenn die erde", "wissen animation"],
    "fr": ["et si la terre", "que se passerait il si", "comment ça marche animation 3d", "expliqué animation",
           "et si le soleil", "le saviez vous animation"],
}
FILTRO_CURTOS = "EgQQARgB"   # tipo vídeo + duração < 4 min


def _yt(args, timeout=300):
    r = subprocess.run([YTDLP, "--quiet", "--no-warnings", "--ignore-errors", *args], capture_output=True, text=True, timeout=timeout)
    try:
        return json.loads(r.stdout) if r.stdout.strip() else None
    except json.JSONDecodeError:
        return None


def _busca(q, n=120):
    url = "https://www.youtube.com/results?" + urllib.parse.urlencode({"search_query": q, "sp": FILTRO_CURTOS})
    j = _yt(["--flat-playlist", "--playlist-end", str(n), "-J", url])
    return [e for e in (j or {}).get("entries", []) if e.get("channel_id")]


def descobrir():
    destino = V3 / "canais_candidatos.json"
    canais = json.load(open(destino)) if destino.exists() else {}
    # sementes: os 54 canais faceless da base v2 e os da coleta de momento
    for r in json.load(open(SNAPUGC / "resultados.json")):
        if r.get("canal_id"):
            canais.setdefault(r["canal_id"], {"nome": r["canal"], "fonte": "base_v2", "acertos": 0, "linguas": []})
    tarefas = [(lingua, q) for lingua, qs in BUSCAS.items() for q in qs]
    feitas = set(map(tuple, json.load(open(V3 / "buscas_feitas.json")))) if (V3 / "buscas_feitas.json").exists() else set()
    tarefas = [t for t in tarefas if t not in feitas]
    with ThreadPoolExecutor(4) as ex:
        for (lingua, q), ents in zip(tarefas, ex.map(lambda t: _busca(t[1] + " #shorts"), tarefas)):
            for e in ents:
                if (e.get("duration") or 999) > 180:
                    continue
                c = canais.setdefault(e["channel_id"], {"nome": e.get("channel"), "fonte": "busca", "acertos": 0, "linguas": []})
                c["acertos"] += 1
                if lingua not in c["linguas"]:
                    c["linguas"].append(lingua)
            feitas.add((lingua, q))
            print(f"[{lingua}] {q}: {len(ents)} resultados, {len(canais)} canais", flush=True)
    V3.mkdir(parents=True, exist_ok=True)
    json.dump(canais, open(destino, "w"), ensure_ascii=False, indent=0)
    json.dump(sorted(feitas), open(V3 / "buscas_feitas.json", "w"), ensure_ascii=False)
    print(f"{len(canais)} canais candidatos -> {destino}")


def _aba(cid, limite=600):
    j = _yt(["--flat-playlist", "--playlist-end", str(limite), "-J", f"https://www.youtube.com/channel/{cid}/shorts"], timeout=600)
    if not j:
        return None
    return [{"id": e["id"], "titulo": e.get("title"), "views": e.get("view_count"), "pos": i}
            for i, e in enumerate(j.get("entries") or []) if e.get("id")]


def abas(min_acertos=2):
    canais = json.load(open(V3 / "canais_candidatos.json"))
    ABAS.mkdir(parents=True, exist_ok=True)
    alvo = [c for c, d in canais.items() if (d["fonte"] != "busca" or d["acertos"] >= min_acertos)
            and not (ABAS / f"{c}.jsonl").exists() and not (ABAS / f"{c}.vazio").exists()]
    print(f"{len(alvo)} canais para baixar a aba Shorts", flush=True)
    coletado = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    def um(cid):
        lst = _aba(cid)
        if not lst or len(lst) < 15:      # canal com poucos Shorts não tem régua
            (ABAS / f"{cid}.vazio").write_text(str(len(lst or [])))
            return cid, 0
        with open(ABAS / f"{cid}.jsonl", "w") as f:
            for x in lst:
                f.write(json.dumps({**x, "coletado": coletado}, ensure_ascii=False) + "\n")
        return cid, len(lst)

    with ThreadPoolExecutor(6) as ex:
        for i, (cid, n) in enumerate(ex.map(um, alvo), 1):
            print(f"{i}/{len(alvo)} {cid} {canais[cid]['nome']}: {n} Shorts", flush=True)


def _cliente():
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from googleapiclient.discovery import build
    token = TOKENS / "simulando.analise.json"
    creds = Credentials.from_authorized_user_file(str(token))
    if not creds.valid:
        creds.refresh(Request())   # renova só em memória; o arquivo do token continua sendo do painel de análise
    return build("youtube", "v3", credentials=creds, cache_discovery=False)


def api():
    yt = _cliente()
    destino = V3 / "videos_api.jsonl"
    feitos = set()
    if destino.exists():
        feitos = {json.loads(l)["id"] for l in open(destino) if l.strip()}
    ids, canais_ids = [], set()
    for arq in sorted(ABAS.glob("*.jsonl")):
        canais_ids.add(arq.stem)
        ids += [json.loads(l)["id"] for l in open(arq) if l.strip()]
    ids = [i for i in dict.fromkeys(ids) if i not in feitos]
    print(f"{len(ids)} vídeos para enriquecer ({len(feitos)} já feitos)", flush=True)
    with open(destino, "a") as f:
        for k in range(0, len(ids), 50):
            lote = ids[k:k + 50]
            for tentativa in range(5):
                try:
                    resp = yt.videos().list(part="snippet,contentDetails,statistics", id=",".join(lote), maxResults=50).execute()
                    break
                except Exception as e:  # noqa: BLE001
                    print("erro na API:", str(e)[:200], flush=True)
                    time.sleep(5 * (tentativa + 1))
            else:
                continue
            achados = set()
            for it in resp.get("items", []):
                sn, cd, st = it["snippet"], it["contentDetails"], it.get("statistics", {})
                achados.add(it["id"])
                f.write(json.dumps({
                    "id": it["id"], "canal_id": sn["channelId"], "canal": sn.get("channelTitle"), "publicado": sn["publishedAt"],
                    "titulo": sn.get("title"), "descricao": sn.get("description"), "tags": sn.get("tags") or [],
                    "categoria": sn.get("categoryId"), "lingua_audio": sn.get("defaultAudioLanguage"), "lingua": sn.get("defaultLanguage"),
                    "duracao": cd.get("duration"), "views": int(st["viewCount"]) if "viewCount" in st else None,
                    "likes": int(st["likeCount"]) if "likeCount" in st else None,
                    "comentarios": int(st["commentCount"]) if "commentCount" in st else None,
                    "coletado": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}, ensure_ascii=False) + "\n")
            for i in set(lote) - achados:   # removido ou privado: grava para não pedir de novo
                f.write(json.dumps({"id": i, "indisponivel": True}) + "\n")
            f.flush()
            if (k // 50) % 20 == 0:
                print(f"{k + len(lote)}/{len(ids)}", flush=True)
    # canais
    canais_ids = sorted(canais_ids)
    saida = {}
    for k in range(0, len(canais_ids), 50):
        resp = yt.channels().list(part="snippet,statistics", id=",".join(canais_ids[k:k + 50]), maxResults=50).execute()
        for it in resp.get("items", []):
            st = it.get("statistics", {})
            saida[it["id"]] = {"nome": it["snippet"]["title"], "criado": it["snippet"].get("publishedAt"),
                               "pais": it["snippet"].get("country"), "inscritos": int(st.get("subscriberCount", 0) or 0),
                               "videos": int(st.get("videoCount", 0) or 0), "views_total": int(st.get("viewCount", 0) or 0)}
    json.dump(saida, open(V3 / "canais_api.json", "w"), ensure_ascii=False, indent=0)
    print(f"{len(saida)} canais com estatísticas")


def capas():
    CAPAS.mkdir(parents=True, exist_ok=True)
    ids = []
    for arq in sorted(ABAS.glob("*.jsonl")):
        ids += [json.loads(l)["id"] for l in open(arq) if l.strip()]
    ids = [i for i in dict.fromkeys(ids) if not (CAPAS / f"{i}.jpg").exists() and not (CAPAS / f"{i}.falhou").exists()]
    print(f"{len(ids)} capas para baixar", flush=True)

    def um(vid):
        for nome in ("hqdefault.jpg", "oar2.jpg"):   # hqdefault: 480x360 (~11 KB) com o Short no centro; basta para o CLIP
            try:
                dados = urllib.request.urlopen(f"https://i.ytimg.com/vi/{vid}/{nome}", timeout=20).read()
                if len(dados) > 2000:
                    (CAPAS / f"{vid}.jpg").write_bytes(dados)
                    return True
            except Exception:  # noqa: BLE001
                continue
        (CAPAS / f"{vid}.falhou").write_text("")
        return False

    with ThreadPoolExecutor(28) as ex:
        ok = 0
        for i, r in enumerate(ex.map(um, ids), 1):
            ok += r
            if i % 2000 == 0:
                print(f"{i}/{len(ids)} capas ({ok} ok)", flush=True)
    print(f"capas: {ok} de {len(ids)}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("etapa", choices=["descobrir", "abas", "api", "capas", "tudo"])
    a = p.parse_args()
    etapas = {"descobrir": [descobrir], "abas": [abas], "api": [api], "capas": [capas],
              "tudo": [descobrir, abas, api, capas]}[a.etapa]
    for fn in etapas:
        fn()
    sys.exit(0)
