"""Figuras e tabelas do v3 a partir de reports/v3/validacao.json e data/v3/oof.parquet.

Uso: .venv/bin/python -m src.v3.relatorio   → reports/v3/figuras/*.png e reports/v3/tabelas/*.csv
"""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.v3.config import RELATORIO, V3  # noqa: E402

SUPERFICIE, TEXTO, TEXTO2, GRADE = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SERIE1, SERIE2 = "#2a78d6", "#eb6834"
FIG = RELATORIO / "figuras"
TAB = RELATORIO / "tabelas"


def _eixo(ax, titulo, xl, yl):
    ax.set_facecolor(SUPERFICIE)
    ax.set_title(titulo, loc="left", color=TEXTO, fontsize=12, pad=10)
    ax.set_xlabel(xl, color=TEXTO2)
    ax.set_ylabel(yl, color=TEXTO2)
    ax.tick_params(colors=TEXTO2)
    ax.grid(True, color=GRADE, linewidth=0.8)
    for s in ax.spines.values():
        s.set_visible(False)


def precisao_por_cobertura(o):
    """Dos X% de Shorts com nota mais alta, quantos estouraram — modelo completo x só contexto do canal."""
    y = o["viral"].to_numpy()
    fr = np.array([0.002, 0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0])
    linhas = []
    for col, nome in (("p_tudo", "modelo completo"), ("p_canal", "só contexto do canal")):
        if col not in o:
            continue
        ordem = np.argsort(-o[col].to_numpy())
        for f in fr:
            k = max(1, int(round(f * len(y))))
            linhas.append({"modelo": nome, "cobertura": f, "precisao": y[ordem[:k]].mean(), "recall": y[ordem[:k]].sum() / y.sum()})
    t = pd.DataFrame(linhas)
    t.to_csv(TAB / "precisao_por_cobertura.csv", index=False)
    fig, ax = plt.subplots(figsize=(7.5, 4.2), facecolor=SUPERFICIE)
    for nome, cor in (("modelo completo", SERIE1), ("só contexto do canal", SERIE2)):
        d = t[t["modelo"] == nome]
        if len(d):
            ax.plot(d["cobertura"] * 100, d["precisao"] * 100, color=cor, linewidth=2, marker="o", markersize=5, label=nome)
    ax.axhline(y.mean() * 100, color=TEXTO2, linewidth=1, linestyle="--")
    ax.text(0.2, y.mean() * 100 + 0.8, f"média da base: {y.mean():.1%}".replace(".", ","), color=TEXTO2, ha="left",
            fontsize=9)
    ax.set_xscale("log")
    ax.set_xticks([0.2, 0.5, 1, 2, 5, 10, 20, 50, 100])
    ax.set_xticklabels(["0,2%", "0,5%", "1%", "2%", "5%", "10%", "20%", "50%", "100%"])
    _eixo(ax, "Dos Shorts com nota mais alta, quantos estouraram", "fração dos Shorts marcados (maiores notas)",
          "estouraram (%)")
    ax.legend(frameon=False, labelcolor=TEXTO2)
    fig.tight_layout()
    fig.savefig(FIG / "precisao_por_cobertura.png", dpi=160)
    plt.close(fig)


def calibracao(o):
    """Calibração honesta: a isotônica de cada dobra é ajustada nas outras 4 e medida nela."""
    from sklearn.isotonic import IsotonicRegression
    y = o["viral"].to_numpy()
    s, d = o["p_tudo"].to_numpy(), o["dobra"].to_numpy()
    p = np.zeros(len(y))
    for k in np.unique(d):
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(s[d != k], y[d != k])
        p[d == k] = iso.predict(s[d == k])
    q = pd.qcut(p, 10, labels=False, duplicates="drop")
    t = pd.DataFrame({"q": q, "p": p, "y": y}).groupby("q").agg(prevista=("p", "mean"), real=("y", "mean"), n=("y", "size"))
    t.to_csv(TAB / "calibracao_decis.csv")
    fig, ax = plt.subplots(figsize=(5.2, 4.6), facecolor=SUPERFICIE)
    m = max(t["prevista"].max(), t["real"].max()) * 100 * 1.1
    ax.plot([0, m], [0, m], color=TEXTO2, linewidth=1, linestyle="--")
    ax.plot(t["prevista"] * 100, t["real"] * 100, color=SERIE1, linewidth=2, marker="o", markersize=6)
    _eixo(ax, "Calibração (decis; ajustada em 4 dobras, medida na 5ª)", "chance prevista (%)", "estouraram de fato (%)")
    fig.tight_layout()
    fig.savefig(FIG / "calibracao.png", dpi=160)
    plt.close(fig)


def por_dispersao(o):
    """Taxa de estouro por quintil de dispersão do canal (p90 - mediana do log das views dos 30 Shorts anteriores)."""
    b = pd.read_parquet(V3 / "base_v3.parquet", columns=["id", "ctx_p90_rel"])
    d = o[["id", "viral"]].merge(b, on="id", how="left").dropna(subset=["ctx_p90_rel"])
    d["q"] = pd.qcut(d["ctx_p90_rel"], 5, labels=False)
    t = d.groupby("q").agg(p90_rel_min=("ctx_p90_rel", "min"), p90_rel_max=("ctx_p90_rel", "max"),
                           taxa_viral=("viral", "mean"), n=("viral", "size"))
    t.to_csv(TAB / "viral_por_dispersao.csv")
    rot = ["1º quinto\n(canais regulares)", "2º", "3º", "4º", "5º quinto\n(canais irregulares)"]
    fig, ax = plt.subplots(figsize=(7.5, 4.2), facecolor=SUPERFICIE)
    x = np.arange(len(t))
    ax.bar(x, t["taxa_viral"] * 100, width=0.6, color=SERIE1, zorder=3)
    for xi, v in zip(x, t["taxa_viral"]):
        ax.text(xi, v * 100 + 0.4, f"{v:.1%}".replace(".", ","), ha="center", color=TEXTO, fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(rot[:len(t)])
    _eixo(ax, "Quanto mais irregular o canal, mais Shorts estouram", "dispersão das views dos 30 Shorts anteriores",
          "estouraram (%)")
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(FIG / "viral_por_dispersao.png", dpi=160)
    plt.close(fig)
    return t


def ablacao(v):
    linhas = [{"conjunto": k, "auc": r["auc"], "ap": r["ap"], "auc_dentro_canal": r.get("auc_dentro_canal"),
               "prec_top1%": r["prec_top1%"], "prec_top5%": r["prec_top5%"]}
              for k, r in v.items() if isinstance(r, dict) and "auc" in r and k != "teste_temporal"]
    t = pd.DataFrame(linhas)
    t.to_csv(TAB / "ablacao.csv", index=False)
    return t


if __name__ == "__main__":
    FIG.mkdir(parents=True, exist_ok=True)
    TAB.mkdir(parents=True, exist_ok=True)
    v = json.load(open(RELATORIO / "validacao.json"))
    o = pd.read_parquet(V3 / "oof.parquet")
    precisao_por_cobertura(o)
    calibracao(o)
    print(por_dispersao(o).round(3).to_string())
    print(ablacao(v).round(3).to_string(index=False))
    pd.DataFrame(v["tudo"]["decisao_viral"]).to_csv(TAB / "decisao_viral.csv", index=False)
    print(pd.DataFrame(v["tudo"]["decisao_viral"]).round(3).to_string(index=False))
