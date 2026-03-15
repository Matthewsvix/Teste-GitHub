# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 14:19:30 2026

@author: matth
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.stats import chi2
import pandas as pd

# 1. Carregando os dados
caminho_arquivo = r"C:\Users\matth\OneDrive\Área de Trabalho\Pesquisas ATR-FTIR\Alex - Cancer de mama\Dados alex.xlsx"
nome_aba = "Gene"
df = pd.read_excel(caminho_arquivo, sheet_name=nome_aba)

# 1. Separando variáveis e classes (Sed vs Treinado)
X = df.iloc[:, 2:]
y = df.iloc[:, 1]
nomes_variaveis = X.columns
samples = df.iloc[:,0]

#%% Testando github

#Fazendo alterações no arquivo
x = 2



#%% Análise de componentes principais (PCA)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from matplotlib.patches import Ellipse

# =========================
# CONFIGURAÇÕES (mude aqui)
# =========================
PC_X = 1         # PC no eixo X (1 = PC1, 2 = PC2, ...)
PC_Y = 2             # PC no eixo Y
N_COMPONENTES = 10       # PCA total

MOSTRAR_ELIPSE = False
NIVEL_ELIPSE = 0.95      # 0.95, 0.99, etc.
ALPHA_ELIPSE = 0.20

MOSTRAR_NOMES = False   # escreve texto fixo em cada ponto
FONTE_NOMES = 20
DESLOC_X = 0.02          # deslocamento do texto (em unidades do score)
DESLOC_Y = 0.02

MOSTRAR_HOVER = False    # mostra nome ao passar o mouse (recomendado)
TAMANHO_PONTO = 350
ALPHA_PONTO = 0.70

CLASSES = ['CTRL', 'CM']                 # ordem precisa bater com os números em y (0,1,...)
CORES  = ['blue', 'red']                 # mesma ordem de CLASSES

FIGSIZE = (12, 10)
FONTE_EIXOS = 26
FONTE_TICKS = 26
FONTE_LEGENDA = 26
MARKER_LEGENDA = 16

# =========================
# DADOS (ajuste se precisar)
# =========================
# X = ...           # seus espectros (amostras x variáveis)
# y = ...           # rótulos numéricos (0,1,...) com mesmo tamanho de X
# df = ...          # seu dataframe com nomes na primeira coluna
samples = df.iloc[:, 0].astype(str).values

# =========================
# 1) Pré-processamento
# =========================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# =========================
# 2) PCA
# =========================
pca = PCA(n_components=N_COMPONENTES)
scores = pca.fit_transform(X_scaled)

# PCs escolhidas (converter PC1->índice 0)
ix = PC_X - 1
iy = PC_Y - 1
pts = scores[:, [ix, iy]]

# =========================
# 3) Cores por classe
# =========================
num_classes = len(CLASSES)
cores = CORES[:num_classes]
class_to_color = {i: cores[i] for i in range(num_classes)}
y_colors = [class_to_color[int(label)] for label in y]

# =========================
# 4) Função elipse
# =========================
def confidence_ellipse(points_2d, level=0.95, color='gray', alpha=0.2):
    """Ellipse baseada em covariância (2D)."""
    if points_2d.shape[0] < 2:
        return

    cov = np.cov(points_2d, rowvar=False)
    vals, vecs = np.linalg.eig(cov)

    order = np.argsort(vals)[::-1]
    vals = vals[order]
    vecs = vecs[:, order]

    s = -2 * np.log(1 - level)
    width  = 2 * np.sqrt(vals[0] * s)
    height = 2 * np.sqrt(vals[1] * s)

    angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    center = points_2d.mean(axis=0)

    e = Ellipse(xy=center, width=width, height=height, angle=angle,
                facecolor=color, edgecolor='none', alpha=alpha)
    plt.gca().add_patch(e)

# =========================
# 5) Plot
# =========================
plt.figure(figsize=FIGSIZE)

scatter = plt.scatter(
    pts[:, 0],
    pts[:, 1],
    c=y_colors,
    alpha=ALPHA_PONTO,
    edgecolor='none',
    s=TAMANHO_PONTO
)

# Nomes fixos (se quiser)
if MOSTRAR_NOMES:
    for i, nome in enumerate(samples):
        plt.text(
            pts[i, 0] + DESLOC_X,
            pts[i, 1] + DESLOC_Y,
            nome,
            fontsize=FONTE_NOMES,
            ha='left',
            va='bottom'
        )

# Elipses por classe (se quiser)
if MOSTRAR_ELIPSE:
    for class_idx in range(num_classes):
        class_points = pts[np.array(y) == class_idx]
        if class_points.shape[0] > 1:
            confidence_ellipse(
                class_points,
                level=NIVEL_ELIPSE,
                color=cores[class_idx],
                alpha=ALPHA_ELIPSE
            )

# Eixos com variância explicada já usando PC_X/PC_Y
vx = pca.explained_variance_ratio_[ix] * 100
vy = pca.explained_variance_ratio_[iy] * 100
plt.xlabel(f'PC{PC_X} ({vx:.1f}%)', fontsize=FONTE_EIXOS)
plt.ylabel(f'PC{PC_Y} ({vy:.1f}%)', fontsize=FONTE_EIXOS)

plt.xticks(fontsize=FONTE_TICKS)
plt.yticks(fontsize=FONTE_TICKS)

# Legenda
legend_elements = [
    plt.Line2D([0], [0], marker='o', color='w', label=CLASSES[i],
               markerfacecolor=cores[i], markersize=MARKER_LEGENDA)
    for i in range(num_classes)
]
plt.legend(handles=legend_elements, fontsize=FONTE_LEGENDA)

# Hover com nome (recomendado)
if MOSTRAR_HOVER:
    try:
        import mplcursors
        cursor = mplcursors.cursor(scatter, hover=True)

        @cursor.connect("add")
        def _on_add(sel):
            idx = sel.index
            sel.annotation.set_text(samples[idx])
    except Exception as e:
        print("Hover não ativado (instale mplcursors: pip install mplcursors). Erro:", e)

plt.tight_layout()
caminho_arquivo = r'C:\Users\matth\OneDrive\Área de Trabalho\Pesquisas ATR-FTIR\Alex - Cancer de mama\pca_gene.TIF' 
plt.savefig(caminho_arquivo, dpi=600)
plt.show()

#%% Partial Least Squares Driscriminant Analysis (PLS-DA) 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler

# =========================
# CONFIGURAÇÕES (mude aqui)
# =========================
# --- Modelo / componentes ---
N_COMPONENTES = 3              # quantos componentes o PLS vai ajustar
LV_X = 1                       # LV no eixo X (1 = LV1)
LV_Y = 2                       # LV no eixo Y (2 = LV2)

# --- Classes / cores (ordem tem que bater com os números do y: 0,1,...) ---
CLASSES = ['CTRL', 'CM']
CORES = ['blue', 'red']

# --- Figura / estilo geral ---
FIGSIZE = (12, 10)
TAMANHO_PONTO = 350
ALPHA_PONTO = 0.70
FONTE_EIXOS = 26
FONTE_TICKS = 26
FONTE_LEGENDA = 26
MARKER_LEGENDA = 16
MOSTRAR_LINHAS_ZERO = True
ESTILO_LINHA_ZERO = '--'
ESPESSURA_LINHA_ZERO = 1

# --- Nomes das amostras nos pontos ---
# use UM ou os DOIS: texto fixo e/ou hover
MOSTRAR_NOMES = False
FONTE_NOMES = 10
DESLOC_X = 0.02
DESLOC_Y = 0.02

MOSTRAR_HOVER = False  # precisa mplcursors (pip install mplcursors)

# --- Elipse ---
MOSTRAR_ELIPSE = True
NIVEL_ELIPSE = 0.95
ALPHA_ELIPSE = 0.20

# --- Biplot (vetores) ---
MOSTRAR_BIPLOT = True
VIP_CUTOFF = 0.0001             # seleciona variáveis com VIP > cutoff
MAX_VETORES = 40              # limita quantidade de vetores (None = sem limite)
ORDENAR_POR_VIP = True        # se True, pega os maiores VIP primeiro
COR_VETORES = 'gray'
ALPHA_VETORES = 0.80
ESPESSURA_SETA = 1.5
HEAD_WIDTH = 0.10             # tamanho da ponta da seta (em unidades do score)
FONTE_ROTULO_VETOR = 16
COR_ROTULO_VETOR = 'black'

# escala do vetor em relação ao gráfico
FRAÇÃO_DO_GRAFICO = 0.95      # 0.8 = ocupa até 80% do range dos scores
AUTO_SCALE = True             # calcula fator automaticamente
SCALING_FACTOR_MANUAL = 1.0   # usado se AUTO_SCALE=False

# =========================
# DADOS (ajuste se precisar)
# =========================
# X = ...   # DataFrame (amostras x variáveis)
# y = ...   # array/Series numérico 0..k-1 com mesmo n de linhas de X
# df = ...  # se quiser nomes das amostras na primeira coluna
samples = df.iloc[:, 0].astype(str).values  # opcional (para nomes/hover)

# =========================
# PRÉ-PROCESSAMENTO
# =========================
scaler = StandardScaler()
X_norm = scaler.fit_transform(X)

# =========================
# PLS-DA (PLSRegression)
# =========================
plsda = PLSRegression(n_components=N_COMPONENTES)
plsda.fit(X_norm, y)

scores = plsda.x_scores_           # (n_amostras, N_COMPONENTES)
loadings = plsda.x_weights_        # pesos/weights (n_variáveis, N_COMPONENTES)

# Escolha de LVs (1->índice 0)
ix = LV_X - 1
iy = LV_Y - 1
pts = scores[:, [ix, iy]]

# "Variância explicada" aproximada no espaço de scores (igual ao seu código)
explained_var = np.var(scores, axis=0) / np.sum(np.var(scores, axis=0))
lvx_var = explained_var[ix] * 100
lvy_var = explained_var[iy] * 100

# =========================
# FUNÇÕES
# =========================
def confidence_ellipse(points_2d, level=0.95, color='gray', alpha=0.2):
    if points_2d.shape[0] < 2:
        return

    cov = np.cov(points_2d, rowvar=False)
    vals, vecs = np.linalg.eig(cov)

    order = np.argsort(vals)[::-1]
    vals = vals[order]
    vecs = vecs[:, order]

    s = -2 * np.log(1 - level)
    width  = 2 * np.sqrt(vals[0] * s)
    height = 2 * np.sqrt(vals[1] * s)

    angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    center = points_2d.mean(axis=0)

    e = Ellipse(xy=center, width=width, height=height, angle=angle,
                facecolor=color, edgecolor='none', alpha=alpha)
    plt.gca().add_patch(e)

def calculate_vip(pls_model):
    # VIP padrão para PLS com 1 saída (y contínuo ou binário)
    t = pls_model.x_scores_
    w = pls_model.x_weights_
    q = pls_model.y_loadings_.ravel()
    p, h = w.shape
    ssq = np.sum((t ** 2) * (q ** 2), axis=0)
    total_ssq = np.sum(ssq)
    vip = np.zeros((p,))
    for i in range(p):
        vip[i] = np.sqrt(p * np.sum((w[i, :] ** 2) * ssq) / total_ssq)
    return vip

# =========================
# CORES POR CLASSE
# =========================
num_classes = len(CLASSES)
cores = CORES[:num_classes]
class_to_color = {i: cores[i] for i in range(num_classes)}
y = np.array(y).astype(int)  # garantir indexação
y_colors = [class_to_color[int(label)] for label in y]

# =========================
# VIP + seleção de vetores (se biplot ligado)
# =========================
vip_scores = calculate_vip(plsda)
vip_df = pd.DataFrame({'Variavel': X.columns, 'VIP_Score': vip_scores})

if MOSTRAR_BIPLOT:
    vip_mask = vip_scores > VIP_CUTOFF
    idx_sel = np.where(vip_mask)[0]

    if idx_sel.size == 0:
        selected_names = np.array([])
        selected_vecs = np.empty((0, 2))
    else:
        # ordenar por VIP para pegar os top MAX_VETORES
        if ORDENAR_POR_VIP:
            idx_sel = idx_sel[np.argsort(vip_scores[idx_sel])[::-1]]

        if (MAX_VETORES is not None) and (idx_sel.size > MAX_VETORES):
            idx_sel = idx_sel[:MAX_VETORES]

        selected_names = np.array(X.columns)[idx_sel]
        selected_vecs = loadings[idx_sel][:, [ix, iy]]  # pesos nas LVs escolhidas (2D)

    # escala automática para caber no gráfico
    if AUTO_SCALE and selected_vecs.shape[0] > 0:
        max_score_x = np.max(np.abs(pts[:, 0]))
        max_score_y = np.max(np.abs(pts[:, 1]))
        max_vec_x = np.max(np.abs(selected_vecs[:, 0]))
        max_vec_y = np.max(np.abs(selected_vecs[:, 1]))

        # evitar divisão por zero
        eps = 1e-12
        max_vec_x = max(max_vec_x, eps)
        max_vec_y = max(max_vec_y, eps)

        scaling_factor = min(
            FRAÇÃO_DO_GRAFICO * max_score_x / max_vec_x,
            FRAÇÃO_DO_GRAFICO * max_score_y / max_vec_y
        )
    else:
        scaling_factor = SCALING_FACTOR_MANUAL

# =========================
# PLOT
# =========================
plt.figure(figsize=FIGSIZE)

scatter = plt.scatter(
    pts[:, 0], pts[:, 1],
    c=y_colors,
    alpha=ALPHA_PONTO,
    edgecolor='none',
    s=TAMANHO_PONTO
)

# Nomes fixos (opcional)
if MOSTRAR_NOMES:
    for i, nome in enumerate(samples):
        plt.text(
            pts[i, 0] + DESLOC_X,
            pts[i, 1] + DESLOC_Y,
            nome,
            fontsize=FONTE_NOMES,
            ha='left',
            va='bottom'
        )

# Elipses (opcional)
if MOSTRAR_ELIPSE:
    for class_idx in range(num_classes):
        class_points = pts[y == class_idx]
        if class_points.shape[0] > 1:
            confidence_ellipse(
                class_points,
                level=NIVEL_ELIPSE,
                color=cores[class_idx],
                alpha=ALPHA_ELIPSE
            )

# Biplot vetores (opcional)
if MOSTRAR_BIPLOT and selected_vecs.shape[0] > 0:
    for nome, vec in zip(selected_names, selected_vecs):
        v = vec * scaling_factor
        plt.arrow(
            0, 0, v[0], v[1],
            color=COR_VETORES,
            alpha=ALPHA_VETORES,
            linewidth=ESPESSURA_SETA,
            head_width=HEAD_WIDTH,
            length_includes_head=True
        )
        plt.text(
            v[0] * 1.08, v[1] * 1.08, str(nome),
            fontsize=FONTE_ROTULO_VETOR,
            ha='center', va='center',
            color=COR_ROTULO_VETOR
        )

# Linhas em zero
if MOSTRAR_LINHAS_ZERO:
    plt.axhline(0, color='gray', linestyle=ESTILO_LINHA_ZERO, linewidth=ESPESSURA_LINHA_ZERO)
    plt.axvline(0, color='gray', linestyle=ESTILO_LINHA_ZERO, linewidth=ESPESSURA_LINHA_ZERO)

# Eixos
plt.xlabel(f'LV{LV_X} ({lvx_var:.1f}%)', fontsize=FONTE_EIXOS)
plt.ylabel(f'LV{LV_Y} ({lvy_var:.1f}%)', fontsize=FONTE_EIXOS)
plt.xticks(fontsize=FONTE_TICKS)
plt.yticks(fontsize=FONTE_TICKS)

# Legenda customizada (sem cmap)
legend_elements = [
    plt.Line2D([0], [0], marker='o', color='w', label=CLASSES[i],
               markerfacecolor=cores[i], markersize=MARKER_LEGENDA)
    for i in range(num_classes)
]
plt.legend(handles=legend_elements, fontsize=FONTE_LEGENDA, loc='best')

# Hover (opcional)
if MOSTRAR_HOVER:
    try:
        import mplcursors
        cursor = mplcursors.cursor(scatter, hover=True)

        @cursor.connect("add")
        def _on_add(sel):
            sel.annotation.set_text(samples[sel.index])
    except Exception as e:
        print("Hover não ativado (instale mplcursors: pip install mplcursors). Erro:", e)

plt.tight_layout()
caminho_arquivo = r'C:\Users\matth\OneDrive\Área de Trabalho\Pesquisas ATR-FTIR\Alex - Cancer de mama\plsda_gene_biplot.TIF' 
plt.savefig(caminho_arquivo, dpi=600)
plt.show()

# =========================
# VIP dataframe (pra você usar depois)
# =========================
vip_df = vip_df.sort_values('VIP_Score', ascending=False).reset_index(drop=True)
# print(vip_df.head(30))

#%% Volcano plot 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# =========================
# CONFIGURAÇÕES
# =========================

# nomes dos grupos
ROTULO_0 = "CTRL"
ROTULO_1 = "CM"

GENES = None   # None usa todas as colunas de X

# teste estatístico
TESTE = "ttest"       # "ttest" ou "mannwhitney"
TTEST_WELCH = False
ALFA = 0.05

# eixo X
X_AXIS_MODE = "percent"  # "percent" ou "log2fc"

# aparência
FIGSIZE = (12,9)
TAMANHO_PONTO = 350
ALPHA_PONTO = 0.9

COR_NS = "lightgray"
COR_UP = "red"
COR_DOWN = "blue"

# linhas
MOSTRAR_LINHA_P = True
MOSTRAR_LINHA_X0 = True

# centralizar eixo
CENTRALIZAR_ZERO = True
MARGEM_EIXO = 1.2

# anotação
MOSTRAR_ANNOT = True
N_ANNOT = 4
FONTE_ANNOT = 20

# salvar figura
SALVAR = True
CAMINHO_SAIDA = r"C:\Users\matth\OneDrive\Área de Trabalho\Pesquisas ATR-FTIR\Alex - Cancer de mama\volcano_plot_gene_mannwithney.TIFF"
DPI = 600

# =========================
# FUNÇÕES
# =========================

def testar_gene(x0, x1):

    if TESTE == "ttest":
        res = stats.ttest_ind(x1, x0, equal_var=not TTEST_WELCH)
        return res.pvalue

    if TESTE == "mannwhitney":
        res = stats.mannwhitneyu(x1, x0)
        return res.pvalue

def efeito_x(m0, m1):

    if X_AXIS_MODE == "percent":
        return ((m1 - m0) / m0) * 100

    if X_AXIS_MODE == "log2fc":
        return np.log2(m1 / m0)

# =========================
# PREPARO
# =========================

y_arr = np.array(y)

if GENES is None:
    GENES = list(X.columns)

idx0 = y_arr == 0
idx1 = y_arr == 1

# =========================
# CÁLCULO
# =========================

rows = []

for gene in GENES:

    x0 = X.loc[idx0, gene].values
    x1 = X.loc[idx1, gene].values

    m0 = np.mean(x0)
    m1 = np.mean(x1)

    efeito = efeito_x(m0, m1)

    p = testar_gene(x0, x1)

    rows.append([gene, efeito, p])

res = pd.DataFrame(rows, columns=["gene","efeito","p"])

res["neglog10"] = -np.log10(res["p"])

# classificação
res["classe"] = "ns"

res.loc[(res["p"] < ALFA) & (res["efeito"] > 0),"classe"] = "up"
res.loc[(res["p"] < ALFA) & (res["efeito"] < 0),"classe"] = "down"

cores = {
    "ns":COR_NS,
    "up":COR_UP,
    "down":COR_DOWN
}

res["cor"] = res["classe"].map(cores)

# =========================
# PLOT
# =========================

plt.figure(figsize=FIGSIZE)

plt.scatter(
    res["efeito"],
    res["neglog10"],
    c=res["cor"],
    s=TAMANHO_PONTO,
    alpha=ALPHA_PONTO,
    edgecolor="black"
)

# linha p
if MOSTRAR_LINHA_P:
    plt.axhline(-np.log10(ALFA), linestyle="--", color="black")

# linha x=0
if MOSTRAR_LINHA_X0:
    plt.axvline(0, linestyle="--", color="black")

# centralizar eixo
if CENTRALIZAR_ZERO:

    max_abs = np.max(np.abs(res["efeito"]))

    limite = max_abs * MARGEM_EIXO

    plt.xlim(-limite, limite)

# anotações
if MOSTRAR_ANNOT:

    sig = res[res["p"] < ALFA]

    sig = sig.sort_values("p").head(N_ANNOT)

    for _, r in sig.iterrows():

        plt.text(
            r["efeito"],
            r["neglog10"]+0.05,
            r["gene"],
            fontsize=FONTE_ANNOT,
            ha="center"
        )

# labels
plt.xlabel("(%)" if X_AXIS_MODE=="percent" else "log2FC", fontsize=24)
plt.ylabel("-Log10(p)", fontsize=24)

plt.xticks(fontsize=24)
plt.yticks(fontsize=24)

# remover bordas superiores
ax = plt.gca()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()

# salvar
if SALVAR:
    plt.savefig(CAMINHO_SAIDA, dpi=DPI)

plt.show()

#%% PLS-DA (PLSRegression) com Leave-One-Out + ROC/AUC 

import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import LeaveOneOut
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_curve, roc_auc_score

# =========================
# CONFIGURAÇÕES (mude aqui)
# =========================
# Modelo
N_COMPONENTES = 3                # número de componentes do PLS
USAR_SCALER = True                # recomendado
Y_POS = 1                         # classe positiva para ROC/AUC (geralmente 1)

# Curva ROC
FIGSIZE = (8, 7)
ESPESSURA_LINHA = 2.5
MOSTRAR_DIAGONAL = True
ESPESSURA_DIAGONAL = 1.5
ESTILO_DIAGONAL = "--"

# Texto
MOSTRAR_AUC_NO_GRAFICO = True
TAMANHO_FONTE = 14

# Eixos
FONTE_EIXOS = 14
FONTE_TICKS = 12

# Salvar figura
SALVAR = True
CAMINHO_SAIDA = r"C:\Users\matth\OneDrive\Área de Trabalho\Pesquisas ATR-FTIR\Alex - Cancer de mama\curva_roc_gene.TIFF"
DPI = 600

# =========================
# DADOS (assume que X e y já existem)
# =========================
# X: DataFrame (n_amostras x n_genes)
# y: array-like (n_amostras,) com 0/1

X_mat = X.values
y_arr = np.asarray(y).astype(int).ravel()

if X_mat.shape[0] != y_arr.shape[0]:
    raise ValueError(f"X tem {X_mat.shape[0]} linhas, mas y tem {y_arr.shape[0]}.")

# =========================
# PIPELINE
# =========================
steps = []
if USAR_SCALER:
    steps.append(("scaler", StandardScaler()))
steps.append(("pls", PLSRegression(n_components=N_COMPONENTES)))
modelo = Pipeline(steps)

# =========================
# LEAVE-ONE-OUT: predições fora-da-amostra
# =========================
loo = LeaveOneOut()
y_score = np.zeros_like(y_arr, dtype=float)

for train_idx, test_idx in loo.split(X_mat):
    X_tr, X_te = X_mat[train_idx], X_mat[test_idx]
    y_tr = y_arr[train_idx]

    modelo.fit(X_tr, y_tr)
    # PLSRegression retorna valores contínuos (scores) -> bom para ROC
    y_score[test_idx[0]] = float(modelo.predict(X_te).ravel()[0])

# =========================
# ROC + AUC
# =========================
# Se a classe positiva for 1, está ok.
# Se você quiser inverter, basta trocar Y_POS e (opcionalmente) transformar score.
if Y_POS == 1:
    scores = y_score
    y_true = y_arr
else:
    # transforma para manter "positivo" como Y_POS
    # (equivalente a inverter rótulos)
    y_true = (y_arr == Y_POS).astype(int)
    scores = y_score

fpr, tpr, _ = roc_curve(y_true, scores, pos_label=1)
auc = roc_auc_score(y_true, scores)

# =========================
# PLOT
# =========================
plt.figure(figsize=FIGSIZE)
ax = plt.gca()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.plot(fpr, tpr, linewidth=ESPESSURA_LINHA)

if MOSTRAR_DIAGONAL:
    plt.plot([0, 1], [0, 1], linestyle=ESTILO_DIAGONAL, linewidth=ESPESSURA_DIAGONAL)

plt.xlabel("False Positive Rate", fontsize=FONTE_EIXOS)
plt.ylabel("True Positive Rate", fontsize=FONTE_EIXOS)
plt.xticks(fontsize=FONTE_TICKS)
plt.yticks(fontsize=FONTE_TICKS)

if MOSTRAR_AUC_NO_GRAFICO:
    plt.text(0.60, 0.10, f"AUC = {auc:.3f}", transform=ax.transAxes, fontsize=TAMANHO_FONTE)

plt.tight_layout()

if SALVAR:
    plt.savefig(CAMINHO_SAIDA, dpi=DPI, bbox_inches="tight")

plt.show()

print(f"AUC (LOO): {auc:.6f}")
































