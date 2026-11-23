"""
Clustering Kualitas Udara Benowo, Surabaya (sheet 2025)
Metode : K-Means (utama) dan DBSCAN (pembanding)

Cara pakai:
    pip install pandas numpy scikit-learn matplotlib openpyxl
    python clustering_kualitas_udara_benowo_2025.py [path_file.xlsx]

Jika path tidak diberikan, dipakai AirQuality_Benowo_xv6vmsv.xlsx di folder yang sama.
Semua gambar (PNG) dan tabel (CSV) disimpan di folder "hasil_clustering".
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

# ----------------------------------------------------------------------------
# 0. Pengaturan
# ----------------------------------------------------------------------------
FILE = sys.argv[1] if len(sys.argv) > 1 else "AirQuality_Benowo_xv6vmsv.xlsx"
SHEET = "2025"
OUT = "hasil_clustering"
os.makedirs(OUT, exist_ok=True)

FITUR = ["PM2.5 (ug/m3)", "PM10 (ug/m3)", "CO2 (ppm)",
         "Temperature (Celsius)", "Humidity (%)"]
K_FINAL = 3
SEED = 42
WARNA = {"Klaster 1": "#2a9d8f", "Klaster 2": "#e9c46a", "Klaster 3": "#e76f51"}


def simpan_gambar(nama):
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, nama), dpi=150)
    plt.close()


# ----------------------------------------------------------------------------
# 1. Baca data
# ----------------------------------------------------------------------------
df = pd.read_excel(FILE, sheet_name=SHEET)
n_mentah = len(df)
df["dt"] = pd.to_datetime(df["Datetime_start"])
print(f"Data mentah sheet {SHEET}: {n_mentah} baris, "
      f"{df['dt'].min()} s.d. {df['dt'].max()}")

# ----------------------------------------------------------------------------
# 2. Preprocessing
# ----------------------------------------------------------------------------
# 2a. Hapus duplikat stempel waktu
df = df.drop_duplicates("dt").sort_values("dt")
n_dedup = len(df)

# 2b. Korelasi antar-variabel (dasar seleksi fitur)
kolom_korelasi = ["AQI US", "AQI CN", "PM2.5 (ug/m3)", "PM10 (ug/m3)", "PM1 (ug/m3)",
                  "CO2 (ppm)", "Temperature (Celsius)", "Temperature (Fahrenheit)",
                  "Humidity (%)"]
df[kolom_korelasi].corr().round(2).to_csv(os.path.join(OUT, "korelasi.csv"))

# 2c. Outlier: buang baris dengan PM2.5 atau PM10 di atas Q3 + 3*IQR
mask = pd.Series(True, index=df.index)
for kol in ["PM2.5 (ug/m3)", "PM10 (ug/m3)"]:
    q1, q3 = df[kol].quantile([0.25, 0.75])
    batas = q3 + 3 * (q3 - q1)
    mask &= df[kol] <= batas
    print(f"Batas outlier {kol}: {batas:.1f}")
dc = df[mask].copy()
n_akhir = len(dc)
print(f"Setelah hapus duplikat: {n_dedup} | setelah outlier: {n_akhir} "
      f"({n_dedup - n_akhir} baris dibuang)")

# 2d. Standarisasi
X = StandardScaler().fit_transform(dc[FITUR])

# ----------------------------------------------------------------------------
# 3. Menentukan jumlah klaster (Elbow, Silhouette, DBI, CH)
# ----------------------------------------------------------------------------
rs = np.random.RandomState(0)
idx = rs.choice(len(X), 6000, replace=False)   # sampel untuk silhouette
ks = list(range(2, 9))
hasil = []
for k in ks:
    km = KMeans(k, n_init=10, random_state=SEED).fit(X)
    hasil.append({
        "k": k,
        "inertia": km.inertia_,
        "silhouette": silhouette_score(X[idx], km.labels_[idx]),
        "DBI": davies_bouldin_score(X, km.labels_),
        "CH": calinski_harabasz_score(X, km.labels_),
    })
ev = pd.DataFrame(hasil)
ev.to_csv(os.path.join(OUT, "evaluasi_k.csv"), index=False)
print("\nEvaluasi jumlah klaster:\n", ev.round(3).to_string(index=False))

fig, ax = plt.subplots(2, 2, figsize=(10, 7))
panel = [("inertia", "Metode Elbow (inertia)", "C0"),
         ("silhouette", "Silhouette Score (makin besar makin baik)", "C1"),
         ("DBI", "Davies-Bouldin Index (makin kecil makin baik)", "C2"),
         ("CH", "Calinski-Harabasz Index (makin besar makin baik)", "C3")]
for a, (kol, judul, warna) in zip(ax.ravel(), panel):
    a.plot(ks, ev[kol], "o-", c=warna)
    a.axvline(K_FINAL, ls="--", c="gray", lw=1)
    a.set(title=judul, xlabel="Jumlah klaster (k)")
simpan_gambar("gambar1_evaluasi_k.png")

# ----------------------------------------------------------------------------
# 4. K-Means final
# ----------------------------------------------------------------------------
km = KMeans(K_FINAL, n_init=10, random_state=SEED).fit(X)
dc["c"] = km.labels_
# Beri nama klaster berurutan menurut rata-rata PM2.5 (rendah -> tinggi)
urutan = dc.groupby("c")["PM2.5 (ug/m3)"].mean().sort_values().index.tolist()
nama = {c: f"Klaster {i + 1}" for i, c in enumerate(urutan)}
dc["klaster"] = dc["c"].map(nama)

profil = dc.groupby("klaster")[FITUR].mean().round(1)
profil["n"] = dc.groupby("klaster").size()
profil["persen"] = (profil["n"] / len(dc) * 100).round(1)
profil.to_csv(os.path.join(OUT, "profil_klaster.csv"))
print("\nProfil klaster:\n", profil.to_string())

# ----------------------------------------------------------------------------
# 4b. Interpretasi dengan kategori AQI US (AQI TIDAK dipakai sebagai fitur;
#     hanya untuk memberi label pada klaster setelah terbentuk)
# ----------------------------------------------------------------------------
kategori = pd.cut(dc["AQI US"], bins=[-1, 50, 100, 150, 200, 300, 10000],
                  labels=["Baik", "Sedang", "Tidak sehat (sensitif)", "Tidak sehat",
                          "Sangat tidak sehat", "Berbahaya"])
aqi_tabel = pd.crosstab(dc["klaster"], kategori, normalize="index").mul(100).round(1)
aqi_tabel.insert(0, "AQI_US_rata2", dc.groupby("klaster")["AQI US"].mean().round(1))
aqi_tabel.to_csv(os.path.join(OUT, "kategori_aqi_per_klaster.csv"))
print("\nKategori AQI US per klaster (%):\n", aqi_tabel.to_string())

# Heatmap profil (z-score)
z = (pd.DataFrame(X, columns=FITUR).assign(klaster=dc["klaster"].values)
     .groupby("klaster").mean())
fig, ax = plt.subplots(figsize=(6.5, 2.8))
im = ax.imshow(z.values, cmap="RdYlBu_r", vmin=-2, vmax=2)
ax.set_xticks(range(5))
ax.set_xticklabels(["PM2.5", "PM10", "CO2", "Suhu", "Kelembapan"])
ax.set_yticks(range(3))
ax.set_yticklabels(z.index)
for i in range(3):
    for j in range(5):
        ax.text(j, i, f"{z.values[i, j]:.2f}", ha="center", va="center", fontsize=9)
plt.colorbar(im, ax=ax, label="z-score")
ax.set_title("Profil rata-rata klaster (data terstandarisasi)")
simpan_gambar("gambar2_heatmap_profil.png")

# PCA
pca = PCA(2).fit(X)
Z = pca.transform(X)
ev_pca = pca.explained_variance_ratio_
print(f"\nPCA: PC1 {ev_pca[0]*100:.1f}% + PC2 {ev_pca[1]*100:.1f}% = {ev_pca.sum()*100:.1f}%")
fig, ax = plt.subplots(figsize=(6.5, 4.6))
for nm, warna in WARNA.items():
    m = (dc["klaster"] == nm).values
    ax.scatter(Z[m, 0], Z[m, 1], s=4, alpha=0.35, c=warna, label=nm)
ax.set_xlabel(f"PC1 ({ev_pca[0]*100:.1f}%)")
ax.set_ylabel(f"PC2 ({ev_pca[1]*100:.1f}%)")
ax.legend(markerscale=4)
ax.set_title("Sebaran klaster pada ruang PCA")
simpan_gambar("gambar3_pca.png")

# ----------------------------------------------------------------------------
# 5. Sebaran klaster menurut waktu
# ----------------------------------------------------------------------------
dc["jam"] = dc["dt"].dt.hour
dc["bulan"] = dc["dt"].dt.month


def waktu(h):
    if h >= 18 or h < 6:
        return "Malam (18-05)"
    return "Pagi (06-11)" if h < 12 else "Siang-sore (12-17)"


dc["waktu"] = dc["jam"].map(waktu)
ct_waktu = pd.crosstab(dc["klaster"], dc["waktu"], normalize="index").mul(100).round(1)
ct_bulan = pd.crosstab(dc["bulan"], dc["klaster"], normalize="index").mul(100).round(1)
ct_waktu.to_csv(os.path.join(OUT, "klaster_per_waktu.csv"))
ct_bulan.to_csv(os.path.join(OUT, "klaster_per_bulan.csv"))
print("\nKomposisi waktu (%):\n", ct_waktu.to_string())
print("\nKomposisi per bulan (%):\n", ct_bulan.to_string())

per_jam = pd.crosstab(dc["jam"], dc["klaster"], normalize="index").mul(100)
fig, ax = plt.subplots(figsize=(6.5, 3.4))
for nm, warna in WARNA.items():
    ax.plot(per_jam.index, per_jam[nm], c=warna, label=nm, lw=2)
ax.set(xlabel="Jam", ylabel="% pengamatan", title="Komposisi klaster menurut jam")
ax.set_xticks(range(0, 24, 3))
ax.legend()
simpan_gambar("gambar4_klaster_per_jam.png")

ct_bulan.plot(kind="bar", stacked=True, color=[WARNA[c] for c in ct_bulan.columns],
              figsize=(6.5, 3.4), width=0.8)
plt.xlabel("Bulan (2025)")
plt.ylabel("%")
plt.title("Komposisi klaster per bulan")
plt.xticks(rotation=0)
plt.legend(title="", loc="upper right", fontsize=8)
simpan_gambar("gambar5_klaster_per_bulan.png")

# ----------------------------------------------------------------------------
# 6. Pembanding: DBSCAN
# ----------------------------------------------------------------------------
rs = np.random.RandomState(1)
baris = []
for eps in [0.4, 0.5, 0.6, 0.8]:
    for min_pts in [30, 50]:
        lab = DBSCAN(eps=eps, min_samples=min_pts).fit_predict(X)
        n_kl = len(set(lab)) - (1 if -1 in lab else 0)
        noise = (lab == -1).mean() * 100
        sil = np.nan
        if n_kl > 1:
            ok = np.where(lab != -1)[0]
            ii = rs.choice(ok, min(5000, len(ok)), replace=False)
            sil = silhouette_score(X[ii], lab[ii])
        baris.append({"eps": eps, "minPts": min_pts, "jumlah_klaster": n_kl,
                      "noise_persen": round(noise, 1),
                      "silhouette": round(sil, 3) if sil == sil else None})
db = pd.DataFrame(baris)
db.to_csv(os.path.join(OUT, "dbscan.csv"), index=False)
print("\nDBSCAN:\n", db.to_string(index=False))

# ----------------------------------------------------------------------------
# 7. Simpan data berlabel
# ----------------------------------------------------------------------------
dc[["dt"] + FITUR + ["klaster"]].to_csv(os.path.join(OUT, "data_berlabel.csv"), index=False)
print(f"\nSelesai. Semua keluaran ada di folder '{OUT}/'.")