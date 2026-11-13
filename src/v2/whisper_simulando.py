"""Transcreve localmente (whisper.cpp, large-v3-turbo) os Shorts públicos do Simulando, com tempo por palavra,
e grava no mesmo formato json3 da legenda automática do YouTube: legendas_whisper/<id>.pt-whisper.json3"""
import glob, json, os, subprocess, sys
from pathlib import Path

S, V = Path(sys.argv[1]), Path(sys.argv[2])
OUT = S / "legendas_whisper"
MODELO = os.path.expanduser("~/.cache/whisper-cpp/ggml-large-v3-turbo.bin")
reg = json.load(open(V / "estudio-shorts" / "analise" / "dados" / "videos.json"))["videos"]
for vid, f in reg.items():
    if f.get("canal") != "simulando" or f.get("privacidade") != "public" or not f.get("entrega"):
        continue
    destino = OUT / f"{vid}.pt-whisper.json3"
    if destino.exists():
        continue
    mp4 = glob.glob(str(V / f["entrega"]) + "/*_1080x1920.mp4")
    if not mp4:
        print("sem mp4:", vid, f["entrega"]); continue
    wav = OUT / f"{vid}.wav"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", mp4[0], "-ar", "16000", "-ac", "1", str(wav)], check=True)
    subprocess.run(["whisper-cli", "-m", MODELO, "-l", "pt", "-ml", "1", "-sow", "-oj", "-np", "-of", str(OUT / vid), str(wav)],
                   check=True, capture_output=True)
    segs = json.load(open(OUT / f"{vid}.json"))["transcription"]
    eventos = []
    for s in segs:
        txt = s["text"].strip()
        if not txt:
            continue
        a, b = s["offsets"]["from"], s["offsets"]["to"]
        eventos.append({"tStartMs": a, "dDurationMs": max(0, b - a), "segs": [{"utf8": txt, "tOffsetMs": 0}]})
    json.dump({"events": eventos, "fonte": "whisper.cpp large-v3-turbo"}, open(destino, "w"), ensure_ascii=False)
    wav.unlink(); (OUT / f"{vid}.json").unlink()
    print(vid, len(eventos), "palavras", flush=True)
print("whisper fim")
