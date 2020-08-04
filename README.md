# ALGORITMA MACHINE LEARNING : USECASE 2 - FRAUD DETECTION <a class='tocSkip'>

# Background

Problem yang ada yaitu banyak terdapat perubahan pola-pola fraud SLI (panggilan
internasional) baru, sehingga semakin banyak pola fraud yang tidak lagi bisa
dideteksi dengan menggunakan **rule base**. Dengan memanfaatkan capability
AI/ML, maka diharapkan bisa meningkatkan (enhance) **FRAMES**.
![ ](assets/Machine-Learning-Task.png)
Dengan memanfaatkan salah satu dari 3 (tiga) kategori **Machine Learning** yaitu
**Unsupervised Learning** untuk mendeteksi fraud. Hal ini dilakukan karena data
yang digunakan tidak memiliki variabel target, sehingga kita akan membiarkan
mesin yang mempelajari data tersebut. Selain itu tujuan dari analisa ini yaitu
bisa menampilkan informasi tersembunyi dari data yang dapat berguna untuk
mendeteksi anomali data. Model yang digunakan yaitu **Anomaly Detection**.

# Import Libraries

Seperti biasa, sebelum memulai analisa dan modeling, lakukan import beberapa
library terkait yang dibutuhkan untuk dikerjakan pada data.

```{.python .input  n=2}
# Data Analysis
import pandas as pd
import numpy as np
from joblib import dump, load
pd.set_option('display.max_columns', 10)

# Visualization
import matplotlib.pyplot as plt
import plotly
import plotly.express as px
from pylab import rcParams
plt.style.use('seaborn')

# Info cell
import time
import warnings
from tqdm import tqdm
warnings.filterwarnings('ignore')

# Modeling
from sklearn import preprocessing, svm
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA 
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.manifold import TSNE

# Magic function
%load_ext autoreload
%load_ext autotime
%autoreload 2
%config IPCompleter.greedy=True
%matplotlib inline
```

```{.json .output n=2}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "The autoreload extension is already loaded. To reload it, use:\n  %reload_ext autoreload\nThe autotime extension is already loaded. To reload it, use:\n  %reload_ext autotime\ntime: 31.8 ms\n"
 }
]
```

# Data Load and Understanding

Dataset yang digunakan untuk analisa ada 2 (dua) dengan rentang waktu
pengambilan data transaksi selama 1 (satu) bulan, pada bulan maret. Dataset yang
dimaksud disimpan ke dalam file, dengan keterangan sebagai berikut :
- **fraud_pstn_202003.csv** :  Data kotor transaksi telepon SLI selama 1 (satu)
bulan pada bulan maret 2020.
- **fraud_pst_maret2020_label1.xlsx** : Data fraud transaksi telepon SLI selama
bulan maret dengan rule base.

Dataset memiliki nama kolom dengan deskripsi sebagai berikut :
- `start` : Tanggal dan jam panggilan telepon dimulai
- `end` : Tanggal dan jam panggilan telepon selesai
- `source_num` : Nomor telepon asal
- `dest_num` : Nomor telepon tujuan
- `access_code` : Kode akses yang digunakan untuk panggilan SLI
- `org_dest_num` : Kode kelompok nomor telepon tujuan
- `duration` : Lama panggilan telepon dalam detik
- `dest_country` : Nama negara tujuan telepon
- `dest_country_status` : Status kategori dari negara tujuan telepon (COMMON,
BLACKLIST, dan WHITELIST)

```{.python .input  n=3}
colname = ['start','end','source_num', 'dest_num', 'access_code', 'org_dest_num', 'duration', 'dest_country', 'dest_country_status']
data = pd.read_csv('dataset/fraud_pstn_202003.csv', sep='\t', names=colname, parse_dates=['start', 'end'])
fraud = pd.read_excel('dataset/fraud_pst_maret2020_label1.xlsx')
```

```{.json .output n=3}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "time: 1.25 s\n"
 }
]
```

```{.python .input  n=3}
data.head()
```

```{.json .output n=3}
[
 {
  "data": {
   "text/html": "<div>\n<style scoped>\n    .dataframe tbody tr th:only-of-type {\n        vertical-align: middle;\n    }\n\n    .dataframe tbody tr th {\n        vertical-align: top;\n    }\n\n    .dataframe thead th {\n        text-align: right;\n    }\n</style>\n<table border=\"1\" class=\"dataframe\">\n  <thead>\n    <tr style=\"text-align: right;\">\n      <th></th>\n      <th>start</th>\n      <th>end</th>\n      <th>source_num</th>\n      <th>dest_num</th>\n      <th>access_code</th>\n      <th>org_dest_num</th>\n      <th>duration</th>\n      <th>dest_country</th>\n      <th>dest_country_status</th>\n    </tr>\n  </thead>\n  <tbody>\n    <tr>\n      <th>0</th>\n      <td>2020-03-04 11:18:48</td>\n      <td>2020-03-04 11:20:40</td>\n      <td>315470709</td>\n      <td>61298799842</td>\n      <td>1017.0</td>\n      <td>10176129</td>\n      <td>100.0</td>\n      <td>AUSTRALIA</td>\n      <td>WHITELIST</td>\n    </tr>\n    <tr>\n      <th>1</th>\n      <td>2020-03-04 10:07:10</td>\n      <td>2020-03-04 10:31:58</td>\n      <td>2150862540</td>\n      <td>02129974855</td>\n      <td>NaN</td>\n      <td>21299748</td>\n      <td>1487.0</td>\n      <td>NaN</td>\n      <td>NaN</td>\n    </tr>\n    <tr>\n      <th>2</th>\n      <td>2020-03-04 11:23:04</td>\n      <td>2020-03-04 11:28:34</td>\n      <td>254669100</td>\n      <td>81662238903</td>\n      <td>1017.0</td>\n      <td>10178166</td>\n      <td>319.0</td>\n      <td>JAPAN</td>\n      <td>WHITELIST</td>\n    </tr>\n    <tr>\n      <th>3</th>\n      <td>2020-03-04 10:33:15</td>\n      <td>2020-03-04 10:33:19</td>\n      <td>2130069819</td>\n      <td>41794943375</td>\n      <td>7.0</td>\n      <td>7417949</td>\n      <td>0.0</td>\n      <td>SWITZERLAND</td>\n      <td>BLACKLIST</td>\n    </tr>\n    <tr>\n      <th>4</th>\n      <td>2020-03-04 10:38:14</td>\n      <td>2020-03-04 10:39:23</td>\n      <td>215265506</td>\n      <td>886227990858</td>\n      <td>1017.0</td>\n      <td>10178862</td>\n      <td>67.0</td>\n      <td>TAIWAN</td>\n      <td>WHITELIST</td>\n    </tr>\n  </tbody>\n</table>\n</div>",
   "text/plain": "                start                 end  source_num      dest_num  \\\n0 2020-03-04 11:18:48 2020-03-04 11:20:40   315470709   61298799842   \n1 2020-03-04 10:07:10 2020-03-04 10:31:58  2150862540   02129974855   \n2 2020-03-04 11:23:04 2020-03-04 11:28:34   254669100   81662238903   \n3 2020-03-04 10:33:15 2020-03-04 10:33:19  2130069819   41794943375   \n4 2020-03-04 10:38:14 2020-03-04 10:39:23   215265506  886227990858   \n\n   access_code  org_dest_num  duration dest_country dest_country_status  \n0       1017.0      10176129     100.0    AUSTRALIA           WHITELIST  \n1          NaN      21299748    1487.0          NaN                 NaN  \n2       1017.0      10178166     319.0        JAPAN           WHITELIST  \n3          7.0       7417949       0.0  SWITZERLAND           BLACKLIST  \n4       1017.0      10178862      67.0       TAIWAN           WHITELIST  "
  },
  "execution_count": 3,
  "metadata": {},
  "output_type": "execute_result"
 },
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "time: 46.7 ms\n"
 }
]
```

```{.python .input  n=4}
fraud.head()
```

```{.json .output n=4}
[
 {
  "data": {
   "text/html": "<div>\n<style scoped>\n    .dataframe tbody tr th:only-of-type {\n        vertical-align: middle;\n    }\n\n    .dataframe tbody tr th {\n        vertical-align: top;\n    }\n\n    .dataframe thead th {\n        text-align: right;\n    }\n</style>\n<table border=\"1\" class=\"dataframe\">\n  <thead>\n    <tr style=\"text-align: right;\">\n      <th></th>\n      <th>LAST_UPDATE</th>\n      <th>CALL_DATE</th>\n      <th>source_num</th>\n      <th>dest_num</th>\n      <th>TOTAL_CALL</th>\n      <th>TOTAL_DURATION</th>\n      <th>DESTINATION</th>\n      <th>LAST_TRUNKIN</th>\n      <th>LAST_TRUNKOUT</th>\n      <th>CDRSOURCE</th>\n      <th>TRUNKIN_OWNER</th>\n      <th>TRUNKOUT_OWNER</th>\n    </tr>\n  </thead>\n  <tbody>\n    <tr>\n      <th>0</th>\n      <td>01-MAR-20 16:38:05</td>\n      <td>01-MAR-20 00:00:00</td>\n      <td>2129185200</td>\n      <td>88233011407</td>\n      <td>12</td>\n      <td>5478</td>\n      <td>SATELITE THURAYA</td>\n      <td>BDJK1G</td>\n      <td>GCTHKS</td>\n      <td>GB_JKT</td>\n      <td>TELKOM</td>\n      <td>TELIN HK SILVER</td>\n    </tr>\n    <tr>\n      <th>1</th>\n      <td>01-MAR-20 16:38:05</td>\n      <td>01-MAR-20 00:00:00</td>\n      <td>2129185200</td>\n      <td>88233011410</td>\n      <td>10</td>\n      <td>3716</td>\n      <td>SATELITE THURAYA</td>\n      <td>BDJK1G</td>\n      <td>GCTHKS</td>\n      <td>GB_JKT</td>\n      <td>TELKOM</td>\n      <td>TELIN HK SILVER</td>\n    </tr>\n    <tr>\n      <th>2</th>\n      <td>01-MAR-20 16:38:05</td>\n      <td>01-MAR-20 00:00:00</td>\n      <td>2129185200</td>\n      <td>88236900059</td>\n      <td>7</td>\n      <td>2789</td>\n      <td>SATELITE THURAYA</td>\n      <td>BDJK1G</td>\n      <td>GCTHKS</td>\n      <td>GB_JKT</td>\n      <td>TELKOM</td>\n      <td>TELIN HK SILVER</td>\n    </tr>\n    <tr>\n      <th>3</th>\n      <td>01-MAR-20 16:38:05</td>\n      <td>01-MAR-20 00:00:00</td>\n      <td>2129185200</td>\n      <td>88236900070</td>\n      <td>8</td>\n      <td>3709</td>\n      <td>SATELITE THURAYA</td>\n      <td>BDJK1G</td>\n      <td>GCTHKS</td>\n      <td>GB_JKT</td>\n      <td>TELKOM</td>\n      <td>TELIN HK SILVER</td>\n    </tr>\n    <tr>\n      <th>4</th>\n      <td>01-MAR-20 17:39:54</td>\n      <td>01-MAR-20 00:00:00</td>\n      <td>2129185200</td>\n      <td>8823494016</td>\n      <td>8</td>\n      <td>6863</td>\n      <td>SATELITE THURAYA</td>\n      <td>BDJK1G</td>\n      <td>GCTHKS</td>\n      <td>GB_JKT</td>\n      <td>TELKOM</td>\n      <td>TELIN HK SILVER</td>\n    </tr>\n  </tbody>\n</table>\n</div>",
   "text/plain": "          LAST_UPDATE           CALL_DATE  source_num     dest_num  \\\n0  01-MAR-20 16:38:05  01-MAR-20 00:00:00  2129185200  88233011407   \n1  01-MAR-20 16:38:05  01-MAR-20 00:00:00  2129185200  88233011410   \n2  01-MAR-20 16:38:05  01-MAR-20 00:00:00  2129185200  88236900059   \n3  01-MAR-20 16:38:05  01-MAR-20 00:00:00  2129185200  88236900070   \n4  01-MAR-20 17:39:54  01-MAR-20 00:00:00  2129185200   8823494016   \n\n   TOTAL_CALL  TOTAL_DURATION       DESTINATION LAST_TRUNKIN LAST_TRUNKOUT  \\\n0          12            5478  SATELITE THURAYA       BDJK1G        GCTHKS   \n1          10            3716  SATELITE THURAYA       BDJK1G        GCTHKS   \n2           7            2789  SATELITE THURAYA       BDJK1G        GCTHKS   \n3           8            3709  SATELITE THURAYA       BDJK1G        GCTHKS   \n4           8            6863  SATELITE THURAYA       BDJK1G        GCTHKS   \n\n  CDRSOURCE TRUNKIN_OWNER   TRUNKOUT_OWNER  \n0    GB_JKT        TELKOM  TELIN HK SILVER  \n1    GB_JKT        TELKOM  TELIN HK SILVER  \n2    GB_JKT        TELKOM  TELIN HK SILVER  \n3    GB_JKT        TELKOM  TELIN HK SILVER  \n4    GB_JKT        TELKOM  TELIN HK SILVER  "
  },
  "execution_count": 4,
  "metadata": {},
  "output_type": "execute_result"
 },
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "time: 22.9 ms\n"
 }
]
```

# Exploratory Data Analysis (EDA)

```{.python .input  n=5}
print(data.shape) 
print(fraud.shape)
```

```{.json .output n=5}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "(391948, 9)\n(149, 12)\ntime: 1.95 ms\n"
 }
]
```

## Understand Missing Values

Hal yang pertama adalah membersihkan data dari data-data yang tidak perlu
ataupun NaN data (missing value).

```{.python .input  n=4}
# Copy data ke dataframe ke nama dataframe baru
data_clean = data.copy()
data_clean.isna().sum()
```

```{.json .output n=4}
[
 {
  "data": {
   "text/plain": "start                     0\nend                       0\nsource_num                0\ndest_num                  1\naccess_code            2150\norg_dest_num              0\nduration                  0\ndest_country           7581\ndest_country_status    7581\ndtype: int64"
  },
  "execution_count": 4,
  "metadata": {},
  "output_type": "execute_result"
 },
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "time: 69.4 ms\n"
 }
]
```

### Inspect missing `dest_num`
Melakukan pemeriksaan pada missing value pada kolom `dest_num`, untuk
digolongkan sebagai inputation error. Cari data lain dengan `source_num` yang
sama dengan oberservasi missing value pada kolom `dest_num` tersebut.

```{.python .input  n=7}
cond1 = data_clean['dest_num'].isna()
cond1
source_num_missing_access_code = data_clean[cond1]['source_num'].unique()
data_clean[data_clean['source_num'].isin(source_num_missing_access_code)]
```

```{.json .output n=7}
[
 {
  "data": {
   "text/html": "<div>\n<style scoped>\n    .dataframe tbody tr th:only-of-type {\n        vertical-align: middle;\n    }\n\n    .dataframe tbody tr th {\n        vertical-align: top;\n    }\n\n    .dataframe thead th {\n        text-align: right;\n    }\n</style>\n<table border=\"1\" class=\"dataframe\">\n  <thead>\n    <tr style=\"text-align: right;\">\n      <th></th>\n      <th>start</th>\n      <th>end</th>\n      <th>source_num</th>\n      <th>dest_num</th>\n      <th>access_code</th>\n      <th>org_dest_num</th>\n      <th>duration</th>\n      <th>dest_country</th>\n      <th>dest_country_status</th>\n    </tr>\n  </thead>\n  <tbody>\n    <tr>\n      <th>233123</th>\n      <td>2020-03-16 21:05:36</td>\n      <td>2020-03-16 21:22:56</td>\n      <td>761943117</td>\n      <td>18778277870</td>\n      <td>1017.0</td>\n      <td>10171877</td>\n      <td>1031.0</td>\n      <td>USA</td>\n      <td>WHITELIST</td>\n    </tr>\n    <tr>\n      <th>286070</th>\n      <td>2020-03-16 21:05:05</td>\n      <td>2020-03-16 21:05:10</td>\n      <td>761943117</td>\n      <td>NaN</td>\n      <td>7.0</td>\n      <td>7</td>\n      <td>0.0</td>\n      <td>NaN</td>\n      <td>NaN</td>\n    </tr>\n  </tbody>\n</table>\n</div>",
   "text/plain": "                     start                 end  source_num     dest_num  \\\n233123 2020-03-16 21:05:36 2020-03-16 21:22:56   761943117  18778277870   \n286070 2020-03-16 21:05:05 2020-03-16 21:05:10   761943117          NaN   \n\n        access_code  org_dest_num  duration dest_country dest_country_status  \n233123       1017.0      10171877    1031.0          USA           WHITELIST  \n286070          7.0             7       0.0          NaN                 NaN  "
  },
  "execution_count": 7,
  "metadata": {},
  "output_type": "execute_result"
 },
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "time: 47.5 ms\n"
 }
]
```

Dari hasil yang didapat, dapat diasumsikan bahwa missing value pada kolom
`dest_num` adalah kesalahan data dan dapat dihapus. Tapi terlebih dahulu di cek
keterkaitannya dengan kolom lainnya.

___

### Inspect missing `access_code`
Melakukan pemeriksaan terhadap kolom `dest_num` yang memiliki msising value
terhadap `access_code`, apakah juga berupa inputation error atau memang tidak
memiliki nilai.

```{.python .input  n=8}
cond2 = data_clean['access_code'].isna()
dest_num_missing_country = data_clean[cond2]['dest_num'].unique()
(data_clean[data_clean['dest_num'].isin(dest_num_missing_country)].index == data_clean[cond2].index).mean()
```

```{.json .output n=8}
[
 {
  "data": {
   "text/plain": "1.0"
  },
  "execution_count": 8,
  "metadata": {},
  "output_type": "execute_result"
 },
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "time: 85.4 ms\n"
 }
]
```

Dari hasil yang diberikan di atas, nilai `access_code` yang missing terjadi pada
beberapa nomor `dest_num`, sehingga asumsi bahwa nilai tersebut dikategorikan
sebagai inputation error dapat kita abaikan, dan dapat diekslkusifkan
(diremove).

```{.python .input  n=5}
# Remove missing access_code (berdasarkan hasil di atas)
data_clean.dropna(subset=['access_code'], inplace=True)
```

```{.json .output n=5}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "time: 57.4 ms\n"
 }
]
```

___

### Inspect missing `dest_country`
Periksa apakah missing `dest_country` berimplikasi pada missing
`deset_country_status`. Hasil dibawah menyebutkan bahwa semua missing
`dest_country`, juga missing `dest_country_status`. Lalu putuskan apakah akan
menghapus data dengan missing value tersebut.

```{.python .input  n=10}
cond3 = data_clean['dest_country'].isna()
cond4 = data_clean['dest_country_status'].isna()
(cond3 == cond4).mean()
```

```{.json .output n=10}
[
 {
  "data": {
   "text/plain": "1.0"
  },
  "execution_count": 10,
  "metadata": {},
  "output_type": "execute_result"
 },
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "time: 48.7 ms\n"
 }
]
```

```{.python .input  n=11}
source_num_missing_country = data_clean[cond3 & cond4]['source_num'].unique()
dest_num_missing_country = data_clean[cond3 & cond4]['dest_num'].unique()
```

```{.json .output n=11}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "time: 13.5 ms\n"
 }
]
```

```{.python .input  n=13}
len(dest_num_missing_country), len(source_num_missing_country), len(data_clean[cond3 & cond4])
```

```{.json .output n=13}
[
 {
  "data": {
   "text/plain": "(2143, 2252, 5431)"
  },
  "execution_count": 13,
  "metadata": {},
  "output_type": "execute_result"
 },
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "time: 6.02 ms\n"
 }
]
```

```{.python .input  n=14}
data_clean[data_clean['dest_num'].isin(dest_num_missing_country)].head()
```

```{.json .output n=14}
[
 {
  "data": {
   "text/html": "<div>\n<style scoped>\n    .dataframe tbody tr th:only-of-type {\n        vertical-align: middle;\n    }\n\n    .dataframe tbody tr th {\n        vertical-align: top;\n    }\n\n    .dataframe thead th {\n        text-align: right;\n    }\n</style>\n<table border=\"1\" class=\"dataframe\">\n  <thead>\n    <tr style=\"text-align: right;\">\n      <th></th>\n      <th>start</th>\n      <th>end</th>\n      <th>source_num</th>\n      <th>dest_num</th>\n      <th>access_code</th>\n      <th>org_dest_num</th>\n      <th>duration</th>\n      <th>dest_country</th>\n      <th>dest_country_status</th>\n    </tr>\n  </thead>\n  <tbody>\n    <tr>\n      <th>188</th>\n      <td>2020-03-02 13:01:02</td>\n      <td>2020-03-02 13:01:29</td>\n      <td>2131186485</td>\n      <td>00886986407114</td>\n      <td>1017.0</td>\n      <td>10170088</td>\n      <td>0.0</td>\n      <td>NaN</td>\n      <td>NaN</td>\n    </tr>\n    <tr>\n      <th>210</th>\n      <td>2020-03-02 18:49:59</td>\n      <td>2020-03-02 18:49:59</td>\n      <td>274441100</td>\n      <td>839855808</td>\n      <td>7.0</td>\n      <td>7839855</td>\n      <td>0.0</td>\n      <td>NaN</td>\n      <td>NaN</td>\n    </tr>\n    <tr>\n      <th>218</th>\n      <td>2020-03-03 11:55:13</td>\n      <td>2020-03-03 11:55:50</td>\n      <td>2150842889</td>\n      <td>00886986411353</td>\n      <td>1017.0</td>\n      <td>10170088</td>\n      <td>27.0</td>\n      <td>NaN</td>\n      <td>NaN</td>\n    </tr>\n    <tr>\n      <th>253</th>\n      <td>2020-03-05 16:56:18</td>\n      <td>2020-03-05 16:56:44</td>\n      <td>2131186117</td>\n      <td>00886973029541</td>\n      <td>1017.0</td>\n      <td>10170088</td>\n      <td>0.0</td>\n      <td>NaN</td>\n      <td>NaN</td>\n    </tr>\n    <tr>\n      <th>306</th>\n      <td>2020-03-01 08:24:37</td>\n      <td>2020-03-01 08:24:37</td>\n      <td>361736838</td>\n      <td>894140236</td>\n      <td>7.0</td>\n      <td>7894140</td>\n      <td>0.0</td>\n      <td>NaN</td>\n      <td>NaN</td>\n    </tr>\n  </tbody>\n</table>\n</div>",
   "text/plain": "                  start                 end  source_num        dest_num  \\\n188 2020-03-02 13:01:02 2020-03-02 13:01:29  2131186485  00886986407114   \n210 2020-03-02 18:49:59 2020-03-02 18:49:59   274441100       839855808   \n218 2020-03-03 11:55:13 2020-03-03 11:55:50  2150842889  00886986411353   \n253 2020-03-05 16:56:18 2020-03-05 16:56:44  2131186117  00886973029541   \n306 2020-03-01 08:24:37 2020-03-01 08:24:37   361736838       894140236   \n\n     access_code  org_dest_num  duration dest_country dest_country_status  \n188       1017.0      10170088       0.0          NaN                 NaN  \n210          7.0       7839855       0.0          NaN                 NaN  \n218       1017.0      10170088      27.0          NaN                 NaN  \n253       1017.0      10170088       0.0          NaN                 NaN  \n306          7.0       7894140       0.0          NaN                 NaN  "
  },
  "execution_count": 14,
  "metadata": {},
  "output_type": "execute_result"
 },
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "time: 52.2 ms\n"
 }
]
```

Dari hasil yang diberikan di atas, nilai `dest_country` yang missing terjadi
pada beberapa nomor `dest_num`, sehingga asumsi bahwa nilai tersebut
dikategorikan sebagai inputation error dapat kita abaikan, dan dapat
dieksklusifkan (diremove).

```{.python .input  n=6}
# delete all missing value `dest_country`
data_clean.dropna(subset=['dest_country'], inplace=True)
```

```{.json .output n=6}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "time: 56.4 ms\n"
 }
]
```

```{.python .input  n=16}
print(data_clean.isna().sum())
```

```{.json .output n=16}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "start                  0\nend                    0\nsource_num             0\ndest_num               0\naccess_code            0\norg_dest_num           0\nduration               0\ndest_country           0\ndest_country_status    0\ndtype: int64\ntime: 84.3 ms\n"
 }
]
```

Simpan data clean ke dalam file untuk mempermudah proses di selanjutnya.

```{.python .input  n=7}
data_clean.to_csv('dataset/data_clean.csv', index=False)
```

```{.json .output n=7}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "time: 3.32 s\n"
 }
]
```

# Data Preparation & Data Wrangling

Tahap ini bertujuan untuk membuat data siap diolah dan dikonsumsi oleh model
machine learning. Beberapa faktor yang mungkin dapat dipertimbangkan untuk di
ekstrak dari data adalah sebagai berikut :

- Pengelompokkan data berdasarkan nomor telepon asal dan tanggal
- durasi overlap (jika ada)
- interval antar panggilan telepon untuk nomor telepon asal yang sama
- kardinalitas tetangga nomor pemanggil
- kardinalitas tetangga nomor dipanggil
- durasi panggilan

Untuk kasus klasifikasi dan deteksi anomali, data harus dapat direpresentasikan
sebagai state of data (1 baris sebagai 1 data). Dalam kasus ini, satu data
adalah satu nomor yang menelpon ke nomor tujuan beserta informasi lainnya yang
harus kita agregasikan. Tapi akan disiapkan juga data ready dengan key tidak
hanya nomor telepon asal (`source_num`), sebagai pilihan data ready.

Untuk itu perlu ditambahkan kolom `day` dan `week` untuk menyimpan data tanggal
dari masing-masing transaksi yang nantinya bisa digunakan sebagai pengelompokkan
(grouping) dan key index data.

```{.python .input  n=5}
data_clean = pd.read_csv('dataset/data_clean.csv', parse_dates=['start', 'end'])
```

```{.json .output n=5}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "time: 1.06 s\n"
 }
]
```

```{.python .input  n=6}
data_clean['day'] = data_clean['start'].dt.date
data_clean['day'] = pd.to_datetime(data_clean.day)
data_clean['week'] = data_clean['start'].dt.week
data_clean.head()
```

```{.json .output n=6}
[
 {
  "data": {
   "text/html": "<div>\n<style scoped>\n    .dataframe tbody tr th:only-of-type {\n        vertical-align: middle;\n    }\n\n    .dataframe tbody tr th {\n        vertical-align: top;\n    }\n\n    .dataframe thead th {\n        text-align: right;\n    }\n</style>\n<table border=\"1\" class=\"dataframe\">\n  <thead>\n    <tr style=\"text-align: right;\">\n      <th></th>\n      <th>start</th>\n      <th>end</th>\n      <th>source_num</th>\n      <th>dest_num</th>\n      <th>access_code</th>\n      <th>org_dest_num</th>\n      <th>duration</th>\n      <th>dest_country</th>\n      <th>dest_country_status</th>\n      <th>day</th>\n      <th>week</th>\n    </tr>\n  </thead>\n  <tbody>\n    <tr>\n      <th>0</th>\n      <td>2020-03-04 11:18:48</td>\n      <td>2020-03-04 11:20:40</td>\n      <td>315470709</td>\n      <td>61298799842</td>\n      <td>1017.0</td>\n      <td>10176129</td>\n      <td>100.0</td>\n      <td>AUSTRALIA</td>\n      <td>WHITELIST</td>\n      <td>2020-03-04</td>\n      <td>10</td>\n    </tr>\n    <tr>\n      <th>1</th>\n      <td>2020-03-04 11:23:04</td>\n      <td>2020-03-04 11:28:34</td>\n      <td>254669100</td>\n      <td>81662238903</td>\n      <td>1017.0</td>\n      <td>10178166</td>\n      <td>319.0</td>\n      <td>JAPAN</td>\n      <td>WHITELIST</td>\n      <td>2020-03-04</td>\n      <td>10</td>\n    </tr>\n    <tr>\n      <th>2</th>\n      <td>2020-03-04 10:33:15</td>\n      <td>2020-03-04 10:33:19</td>\n      <td>2130069819</td>\n      <td>41794943375</td>\n      <td>7.0</td>\n      <td>7417949</td>\n      <td>0.0</td>\n      <td>SWITZERLAND</td>\n      <td>BLACKLIST</td>\n      <td>2020-03-04</td>\n      <td>10</td>\n    </tr>\n    <tr>\n      <th>3</th>\n      <td>2020-03-04 10:38:14</td>\n      <td>2020-03-04 10:39:23</td>\n      <td>215265506</td>\n      <td>886227990858</td>\n      <td>1017.0</td>\n      <td>10178862</td>\n      <td>67.0</td>\n      <td>TAIWAN</td>\n      <td>WHITELIST</td>\n      <td>2020-03-04</td>\n      <td>10</td>\n    </tr>\n    <tr>\n      <th>4</th>\n      <td>2020-03-04 10:40:39</td>\n      <td>2020-03-04 10:41:05</td>\n      <td>778424823</td>\n      <td>60377257257</td>\n      <td>7.0</td>\n      <td>7603772</td>\n      <td>4.0</td>\n      <td>MALAYSIA</td>\n      <td>WHITELIST</td>\n      <td>2020-03-04</td>\n      <td>10</td>\n    </tr>\n  </tbody>\n</table>\n</div>",
   "text/plain": "                start                 end  source_num      dest_num  \\\n0 2020-03-04 11:18:48 2020-03-04 11:20:40   315470709   61298799842   \n1 2020-03-04 11:23:04 2020-03-04 11:28:34   254669100   81662238903   \n2 2020-03-04 10:33:15 2020-03-04 10:33:19  2130069819   41794943375   \n3 2020-03-04 10:38:14 2020-03-04 10:39:23   215265506  886227990858   \n4 2020-03-04 10:40:39 2020-03-04 10:41:05   778424823   60377257257   \n\n   access_code  org_dest_num  duration dest_country dest_country_status  \\\n0       1017.0      10176129     100.0    AUSTRALIA           WHITELIST   \n1       1017.0      10178166     319.0        JAPAN           WHITELIST   \n2          7.0       7417949       0.0  SWITZERLAND           BLACKLIST   \n3       1017.0      10178862      67.0       TAIWAN           WHITELIST   \n4          7.0       7603772       4.0     MALAYSIA           WHITELIST   \n\n         day  week  \n0 2020-03-04    10  \n1 2020-03-04    10  \n2 2020-03-04    10  \n3 2020-03-04    10  \n4 2020-03-04    10  "
  },
  "execution_count": 6,
  "metadata": {},
  "output_type": "execute_result"
 },
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "time: 954 ms\n"
 }
]
```

## Inspeksi source
Melakukan wrangling pada level `source_num` secara individual

**Task :** Untuk setiap `source_num`, dapatkan :
- rata-rata interval waktu antar panggilan keluar.
- jumlah durasi panggilan
- jumlah wilayah tujuan (`org_des_num`)

Hint : Gunakan `.shift()` method untuk menghitung selisih waktu panggilan
diakhiri denngan panggilan dimulai berikutnya

## Inspeksi Source Neighbor
Melakukan wrangling pada nomor-nomor yang mirip dengan `source_num`.

**Task** : Dapatkan:
- kardinalitas tetangga (jumlah nomor unik, inklusif)
- kardinalitas nomor yang dipanggil (jumlah `dest_num` unik, inklusif)
- total durasi panggilan

## Data Aggregation
menggabungkan antara :
- feature `source_num` (total durasi, jumlah nomor unik yang dipanggil, jumlah
wilayah yang dituju, rata-rata interval antar panggilan)
- feature tetangga `source_num` (ukuran tetangga, jumlah nomor unik yang
dipanggil, jumlah durasi panggilan)

**Task:** Buatlah sebuah fungsi untuk melakukan wrangling data menjadi data yang
siap kita jadikan sebagai input kedalam machine learning. <br>
Input : Data Clean <br>
Output : Ready-to-feed Data

sample data siap pakai ada pada file `data_ready_sample.csv` dengan bentuk
kurang lebih sebagai berikut (index=`source_num`):

|     source_num |   duration |   dest_num |   org_dest_num |
source_num_nunique |   dest_num_nunique |   total_duration |   avg_interval |
|---------------:|-----------:|-----------:|---------------:|---------------------:|-------------------:|-----------------:|---------------:|
|          21147 |        104 |         18 |             11 |
2 |                 26 |              104 |          30306 |
|          24147 |        954 |        116 |             29 |
1 |                116 |              954 |          21601 |
|         299999 |        236 |        348 |             18 |
1 |                348 |              236 |           6911 |
|         757845 |       1311 |          7 |              7 |
1 |                  7 |             1311 |          45757 |
|        2114000 |       3235 |        156 |             54 |
7 |                177 |             3467 |          16845 |
|        ...     |       ...  |        ... |            ... |
... |                ... |              ... |            ... |
| 21806831903213 |       1087 |          2 |              2 |
1 |                  2 |             1087 |           9922 |
| 62895401351782 |         21 |          2 |              2 |
23 |                 39 |              191 |           5951 |
| 62895401351788 |         76 |          2 |              2 |
23 |                 39 |              191 |          85835 |
| 62895401351813 |          4 |          2 |              2 |
23 |                 39 |              191 |          55327 |
| 62895401351833 |         12 |          2 |              2 |
23 |                 39 |              191 |          74845 |

## Definisi fungsi

Disiapkan fungsi untuk menghitung kardinalitas tetangga berdasarkan key index.

```{.python .input  n=11}
# Fungsi untuk mendapatkan tetangga nomor pemanggil
def find_neighbor(source_num, neighbor, threshold=100):
    return neighbor[abs(neighbor - source_num) <= threshold].unique()
```

```{.json .output n=11}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "time: 998 \u00b5s\n"
 }
]
```

```{.python .input  n=12}
# Fungsi untuk mendapatkan tetangga nomor pemanggil berdasarkan harian data
def find_neighbor_day(source_num, day, neighbor_num, neighbor_day, threshold = 100) :
    cond1 = abs(neighbor_num - source_num) <= threshold
    cond2 = neighbor_day == day
    return neighbor_num[cond1 & cond2].unique()
```

```{.json .output n=12}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "time: 2.99 ms\n"
 }
]
```

```{.python .input  n=13}
# Fungsi untuk mendapatkan tetangga nomor pemanggil berdasarkan harian data dan nomor tujuan yang sama dipanggil oleh source_num
def find_neighbor_day_dest(source_num, day, dest_num, neighbor_num, neighbor_day, neighbor_dest_num, threshold = 100) :
    cond1 = abs(neighbor_num - source_num) <= threshold
    cond2 = neighbor_day == day
    cond3 = neighbor_dest_num == dest_num
    return neighbor_num[cond1 & cond2].unique()
```

```{.json .output n=13}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "time: 996 \u00b5s\n"
 }
]
```

___

### Menggunakan key `source_num`
Dilakukan data wrangling pada  `source_num` sebagai index grouping per
individual nomor telepon asal (`source_num`). Kemudian dilakukan agregasi untuk
setiap `source_num` sebagai berikut :
- Jumlah durasi panggilan telepon (`total_duration` : sum)
- Jumlah panggilan telepon (`total_call` : sum)
- Jumlah nomor tujuan unik yang ditelepon oleh nomor asal (`dest_num` : nunique)
- Jumlah kode akses yang unik (`access_code` : nunique)
- Jumlah wilayah tujuan yang unik (`org_dest_num` : nunique)

```{.python .input  n=28}
data_ready_1 = data_clean.groupby(['source_num']).agg({
    'source_num' : 'count',
    'duration' : 'sum',
    'dest_num' : 'nunique',
    'access_code' : 'nunique',
    'org_dest_num' : 'nunique'
})
data_ready_1.rename(columns={'source_num' : 'total_call',
'duration' : 'total_duration'}, 
inplace=True)
data_ready_1.head()
```

```{.json .output n=28}
[
 {
  "data": {
   "text/html": "<div>\n<style scoped>\n    .dataframe tbody tr th:only-of-type {\n        vertical-align: middle;\n    }\n\n    .dataframe tbody tr th {\n        vertical-align: top;\n    }\n\n    .dataframe thead th {\n        text-align: right;\n    }\n</style>\n<table border=\"1\" class=\"dataframe\">\n  <thead>\n    <tr style=\"text-align: right;\">\n      <th></th>\n      <th>total_call</th>\n      <th>total_duration</th>\n      <th>dest_num</th>\n      <th>access_code</th>\n      <th>org_dest_num</th>\n    </tr>\n    <tr>\n      <th>source_num</th>\n      <th></th>\n      <th></th>\n      <th></th>\n      <th></th>\n      <th></th>\n    </tr>\n  </thead>\n  <tbody>\n    <tr>\n      <th>21123</th>\n      <td>8</td>\n      <td>0.0</td>\n      <td>8</td>\n      <td>1</td>\n      <td>3</td>\n    </tr>\n    <tr>\n      <th>21147</th>\n      <td>18</td>\n      <td>104.0</td>\n      <td>18</td>\n      <td>1</td>\n      <td>11</td>\n    </tr>\n    <tr>\n      <th>24147</th>\n      <td>118</td>\n      <td>954.0</td>\n      <td>116</td>\n      <td>1</td>\n      <td>29</td>\n    </tr>\n    <tr>\n      <th>62078</th>\n      <td>1</td>\n      <td>4062.0</td>\n      <td>1</td>\n      <td>1</td>\n      <td>1</td>\n    </tr>\n    <tr>\n      <th>62147</th>\n      <td>1</td>\n      <td>4.0</td>\n      <td>1</td>\n      <td>1</td>\n      <td>1</td>\n    </tr>\n  </tbody>\n</table>\n</div>",
   "text/plain": "            total_call  total_duration  dest_num  access_code  org_dest_num\nsource_num                                                                 \n21123                8             0.0         8            1             3\n21147               18           104.0        18            1            11\n24147              118           954.0       116            1            29\n62078                1          4062.0         1            1             1\n62147                1             4.0         1            1             1"
  },
  "execution_count": 28,
  "metadata": {},
  "output_type": "execute_result"
 },
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "time: 836 ms\n"
 }
]
```

Selanjutnya dilakukan data wrangling terhadap column `source_num` yang unik,
untuk bisa mendapatkan :
- Jumlah kardinalitas nomor asal (`source_num`) yang mirip atau berdekatan
secara unik.
- Jumlah kardinalitas nomor tujuan yang (`dest_num`) yang mirip atau berdekatan
secara unik ditelepon oleh tetangganya.
- Rata-rata dari interval waktu antar panggilan keluar dari masing-masing nomor
telepon (`source_num`).

Kemudian gabungkan dengan data ready sebelumnya, dan simpan ke dalam file untuk
pemrosesan lebih lanjut.

```{.python .input  n=35}
intervals = []
neighbor_sources = []
neighbor_destinations = []
neighbor_durations = []
for num in tqdm(data_ready_1.iloc[:].index) :
#     print(num)
    df = data_clean[data_clean.source_num == num]
    interval = (df['start'].shift(-1) - df['end']).mean().seconds
    intervals.append(interval)
    neighbor_source = find_neighbor(num, data_clean.source_num, 50)
    neighbor_sources.append(len(neighbor_source))
    neighbor = data_clean[(data_clean.source_num.isin(neighbor_source))]
    neighbor_destinations.append(neighbor.dest_num.nunique())
    neighbor_durations.append(neighbor.duration.sum())
```

```{.json .output n=35}
[
 {
  "name": "stderr",
  "output_type": "stream",
  "text": "100%|\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588| 44227/44227 [08:04<00:00, 91.37it/s]"
 },
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "time: 8min 4s\n"
 },
 {
  "name": "stderr",
  "output_type": "stream",
  "text": "\n"
 }
]
```

```{.python .input  n=37}
data_ready_1['source_num_neighbor_unique'] = neighbor_sources
data_ready_1['dest_num_neighbor_unique'] = neighbor_destinations
data_ready_1['avg_interval'] = intervals
data_ready_1['duration_neighbor'] = neighbor_durations
data_ready_1['avg_interval'] = data_ready_1['avg_interval'].fillna(0)
# data_ready_1.head()
```

```{.json .output n=37}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "time: 41.1 ms\n"
 }
]
```

```{.python .input  n=49}
data_ready_1.to_csv('dataset/fraud_data_ready.csv', index=True)
```

```{.json .output n=49}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "time: 188 ms\n"
 }
]
```

### Menggunakan key `source_num` dan `day`
Dilakukan data wrangling pada  `source_num` dan `day` sebagai index grouping
agar data bisa dilihat harian (`day`) per individual nomor telepon asal
(`source_num`). Kemudian dilakukan agregasi untuk setiap `source_num` dan `day`
sebagai berikut :
- Jumlah durasi panggilan telepon (`total_duration` : sum)
- Jumlah panggilan telepon (`total_call` : sum)
- Jumlah nomor tujuan unik yang ditelepon oleh nomor asal (`dest_num` : nunique)
- Jumlah kode akses yang unik (`access_code` : nunique)
- Jumlah wilayah tujuan yang unik (`org_dest_num` : nunique)

```{.python .input  n=39}
data_ready_2 = data_clean.groupby(['source_num', 'day']).agg({
    'source_num' : 'count',
    'duration' : 'sum',
    'dest_num' : 'nunique',
    'access_code' : 'nunique',
    'org_dest_num' : 'nunique'
})
data_ready_2.rename(columns={'source_num' : 'total_call',
'duration' : 'total_duration'}, 
inplace=True)
data_ready_2.head()
```

```{.json .output n=39}
[
 {
  "data": {
   "text/html": "<div>\n<style scoped>\n    .dataframe tbody tr th:only-of-type {\n        vertical-align: middle;\n    }\n\n    .dataframe tbody tr th {\n        vertical-align: top;\n    }\n\n    .dataframe thead th {\n        text-align: right;\n    }\n</style>\n<table border=\"1\" class=\"dataframe\">\n  <thead>\n    <tr style=\"text-align: right;\">\n      <th></th>\n      <th></th>\n      <th>total_call</th>\n      <th>total_duration</th>\n      <th>dest_num</th>\n      <th>access_code</th>\n      <th>org_dest_num</th>\n    </tr>\n    <tr>\n      <th>source_num</th>\n      <th>day</th>\n      <th></th>\n      <th></th>\n      <th></th>\n      <th></th>\n      <th></th>\n    </tr>\n  </thead>\n  <tbody>\n    <tr>\n      <th rowspan=\"3\" valign=\"top\">21123</th>\n      <th>2020-03-12</th>\n      <td>4</td>\n      <td>0.0</td>\n      <td>4</td>\n      <td>1</td>\n      <td>2</td>\n    </tr>\n    <tr>\n      <th>2020-03-18</th>\n      <td>3</td>\n      <td>0.0</td>\n      <td>3</td>\n      <td>1</td>\n      <td>1</td>\n    </tr>\n    <tr>\n      <th>2020-03-21</th>\n      <td>1</td>\n      <td>0.0</td>\n      <td>1</td>\n      <td>1</td>\n      <td>1</td>\n    </tr>\n    <tr>\n      <th rowspan=\"2\" valign=\"top\">21147</th>\n      <th>2020-03-01</th>\n      <td>1</td>\n      <td>0.0</td>\n      <td>1</td>\n      <td>1</td>\n      <td>1</td>\n    </tr>\n    <tr>\n      <th>2020-03-02</th>\n      <td>4</td>\n      <td>0.0</td>\n      <td>4</td>\n      <td>1</td>\n      <td>1</td>\n    </tr>\n  </tbody>\n</table>\n</div>",
   "text/plain": "                       total_call  total_duration  dest_num  access_code  \\\nsource_num day                                                             \n21123      2020-03-12           4             0.0         4            1   \n           2020-03-18           3             0.0         3            1   \n           2020-03-21           1             0.0         1            1   \n21147      2020-03-01           1             0.0         1            1   \n           2020-03-02           4             0.0         4            1   \n\n                       org_dest_num  \nsource_num day                       \n21123      2020-03-12             2  \n           2020-03-18             1  \n           2020-03-21             1  \n21147      2020-03-01             1  \n           2020-03-02             1  "
  },
  "execution_count": 39,
  "metadata": {},
  "output_type": "execute_result"
 },
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "time: 430 ms\n"
 }
]
```

Selanjutnya dilakukan data wrangling terhadap column `source_num` yang unik
perhari (`day`), untuk bisa mendapatkan :
- Jumlah kardinalitas nomor asal (`source_num`) yang mirip atau berdekatan
secara unik dalam sehari.
- Jumlah kardinalitas nomor tujuan yang (`dest_num`) yang mirip atau berdekatan
secara unik ditelepon oleh tetangga nya dalam sehari.
- Rata-rata dari interval waktu antar panggilan keluar dari masing-masing nomor
telepon (`source_num`) dalam sehari.

Kemudian gabungkan dengan data ready sebelumnya, dan simpan ke dalam file untuk
pemrosesan lebih lanjut.

```{.python .input  n=41}
intervals = []
neighbor_sources = []
neighbor_destinations = []
neighbor_durations = []
for num in tqdm(data_ready_2.iloc[:].index) :
#     print(num)
    df = data_clean[(data_clean.source_num == num[0]) & (data_clean.day == num[1])]
    interval = (df['start'].shift(-1) - df['end']).mean().seconds
    intervals.append(interval)
    neighbor_source = find_neighbor_day(num[0], num[1], data_clean.source_num, data_clean.day, 50)
    neighbor_sources.append(len(neighbor_source))
    neighbor = data_clean[(data_clean.source_num.isin(neighbor_source)) & (data_clean.day == num[1])]
    neighbor_destinations.append(neighbor.dest_num.nunique())
    neighbor_durations.append(neighbor.duration.sum())
```

```{.json .output n=41}
[
 {
  "name": "stderr",
  "output_type": "stream",
  "text": "100%|\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588| 130495/130495 [29:46<00:00, 73.06it/s]"
 },
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "time: 29min 46s\n"
 },
 {
  "name": "stderr",
  "output_type": "stream",
  "text": "\n"
 }
]
```

```{.python .input  n=42}
data_ready_2['source_num_neighbor_unique'] = neighbor_sources
data_ready_2['dest_num_neighbor_unique'] = neighbor_destinations
data_ready_2['avg_interval'] = intervals
data_ready_2['duration_neighbor'] = neighbor_durations
data_ready_2['avg_interval'] = data_ready_2['avg_interval'].fillna(0)
# data_ready_1.head()
```

```{.json .output n=42}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "time: 88.8 ms\n"
 }
]
```

```{.python .input  n=48}
data_ready_2.to_csv('dataset/fraud_data_ready_day.csv', index=True)
```

```{.json .output n=48}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "time: 1.12 s\n"
 }
]
```

___

### Menggunakan key `source_num`, `day`, dan `dest_num`
Dilakukan data wrangling pada  `source_num`, `day`, dan `dest_num` sebagai index
grouping agar data bisa dilihat harian (`day`) per individual nomor telepon asal
(`source_num`) dengan masing-masing nomor tujuan (`dest_num`). Kemudian
dilakukan agregasi untuk setiap `source_num`, `day` dan `dest_num` sebagai
berikut :
- Jumlah durasi panggilan telepon (`total_duration` : sum)
- Jumlah panggilan telepon (`total_call` : sum)
- Jumlah nomor tujuan unik yang ditelepon oleh nomor asal (`dest_num` : nunique)
- Jumlah kode akses yang unik (`access_code` : nunique)
- Jumlah wilayah tujuan yang unik (`org_dest_num` : nunique)

```{.python .input  n=44}
data_ready_3 = data_clean.groupby(['source_num', 'day', 'dest_num']).agg({
    'source_num' : 'count',
    'duration' : 'sum',
    'access_code' : 'nunique',
    'org_dest_num' : 'nunique'
})
data_ready_3.rename(columns={'source_num' : 'total_call',
'duration' : 'total_duration'}, 
inplace=True)
data_ready_3.head()
```

```{.json .output n=44}
[
 {
  "data": {
   "text/html": "<div>\n<style scoped>\n    .dataframe tbody tr th:only-of-type {\n        vertical-align: middle;\n    }\n\n    .dataframe tbody tr th {\n        vertical-align: top;\n    }\n\n    .dataframe thead th {\n        text-align: right;\n    }\n</style>\n<table border=\"1\" class=\"dataframe\">\n  <thead>\n    <tr style=\"text-align: right;\">\n      <th></th>\n      <th></th>\n      <th></th>\n      <th>total_call</th>\n      <th>total_duration</th>\n      <th>access_code</th>\n      <th>org_dest_num</th>\n    </tr>\n    <tr>\n      <th>source_num</th>\n      <th>day</th>\n      <th>dest_num</th>\n      <th></th>\n      <th></th>\n      <th></th>\n      <th></th>\n    </tr>\n  </thead>\n  <tbody>\n    <tr>\n      <th rowspan=\"5\" valign=\"top\">21123</th>\n      <th rowspan=\"4\" valign=\"top\">2020-03-12</th>\n      <th>8613443552056</th>\n      <td>1</td>\n      <td>0.0</td>\n      <td>1</td>\n      <td>1</td>\n    </tr>\n    <tr>\n      <th>8613443552545</th>\n      <td>1</td>\n      <td>0.0</td>\n      <td>1</td>\n      <td>1</td>\n    </tr>\n    <tr>\n      <th>8613444552833</th>\n      <td>1</td>\n      <td>0.0</td>\n      <td>1</td>\n      <td>1</td>\n    </tr>\n    <tr>\n      <th>8615644093456</th>\n      <td>1</td>\n      <td>0.0</td>\n      <td>1</td>\n      <td>1</td>\n    </tr>\n    <tr>\n      <th>2020-03-18</th>\n      <th>6593342553</th>\n      <td>1</td>\n      <td>0.0</td>\n      <td>1</td>\n      <td>1</td>\n    </tr>\n  </tbody>\n</table>\n</div>",
   "text/plain": "                                     total_call  total_duration  access_code  \\\nsource_num day        dest_num                                                 \n21123      2020-03-12 8613443552056           1             0.0            1   \n                      8613443552545           1             0.0            1   \n                      8613444552833           1             0.0            1   \n                      8615644093456           1             0.0            1   \n           2020-03-18 6593342553              1             0.0            1   \n\n                                     org_dest_num  \nsource_num day        dest_num                     \n21123      2020-03-12 8613443552056             1  \n                      8613443552545             1  \n                      8613444552833             1  \n                      8615644093456             1  \n           2020-03-18 6593342553                1  "
  },
  "execution_count": 44,
  "metadata": {},
  "output_type": "execute_result"
 },
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "time: 593 ms\n"
 }
]
```

Selanjutnya dilakukan data wrangling terhadap column `source_num` yang unik
perhari (`day`) untuk masing-masing nomor tujuan (`dest_num`), untuk bisa
mendapatkan :
- Jumlah kardinalitas nomor asal (`source_num`) yang mirip atau berdekatan
secara unik dalam sehari.
- Jumlah kardinalitas nomor tujuan yang (`dest_num`) yang mirip atau berdekatan
secara unik ditelepon oleh tetangga nya dalam sehari.
- Rata-rata dari interval waktu antar panggilan keluar dari masing-masing nomor
telepon (`source_num`) dalam sehari.
- Rata-rata dari interval waktu antar panggilan keluar dari masing-masing nomor
telepon (`source_num`) dalam sehari dengan nomor tujuan yang sama.
- Total panggilan untuk masing-masing nomor asal (`source_num`) dengan kategori
include dan exclude data durasi (`duration`) yang 0 dalam sehari.
- Total durasi untuk masing-masing nomor asal (`source_num`) dengan kategori
include dan exclude data durasi (`duration`) yang 0 dalam sehari.
- Total panggilan untuk masing-masing nomor tujuan (`dest_num`) dengan kategori
include dan exclude data durasi (`duration`) yang 0 dalam sehari.
- Rasio blacklist (`dest_country_status`) untuk masing-masing nomor asal
(`source_num`) dengan kategori include dan exclude data durasi (`duration`) yang
0 dalam sehari.

Kemudian gabungkan dengan data ready sebelumnya, dan simpan ke dalam file untuk
pemrosesan lebih lanjut.

```{.python .input  n=46}
intervals = []
total_duration_source_nums = []
total_duration_source_nums_exc = []
total_call_source_nums = []
total_call_source_nums_exc = []
total_call_dest_nums = []
total_call_dest_nums_exc = []
intervals_all_dest_num = []
ratio_blacklist = []
ratio_blacklist_exc = []
neighbor_sources = []
neighbor_destinations = []
neighbor_durations = []
# for num, row in tqdm(data_ready_2.iterrows()):
for num in tqdm(data_ready_3.iloc[:].index) :
    # print(num)
    df = data_clean[(data_clean.source_num == num[0]) & (data_clean.day == num[1]) & (data_clean.dest_num == num[2])]
    interval = (df['start'].shift(-1) - df['end']).mean().seconds
    intervals.append(interval)
    df = data_clean[(data_clean.source_num == num[0]) & (data_clean.day == num[1])]
    interval = (df['start'].shift(-1) - df['end']).mean().seconds
    intervals_all_dest_num.append(interval)
    all_call_source_num = data_clean[(data_clean.source_num == num[0]) & (data_clean.day == num[1])]
    all_call_source_num_exc = data_clean[(data_clean.source_num == num[0]) & (data_clean.day == num[1]) & (data_clean.duration > 0)]
    total_duration_source_nums.append(all_call_source_num['duration'].sum())
    total_duration_source_nums_exc.append(all_call_source_num_exc['duration'].sum())
    total_call_source_nums.append(len(all_call_source_num))
    total_call_source_nums_exc.append(len(all_call_source_num_exc))
    
    all_call_dest_num = data_clean[(data_clean.dest_num == num[2]) & (data_clean.day == num[1])]
    all_call_dest_num_exc = data_clean[(data_clean.dest_num == num[2]) & (data_clean.day == num[1]) & (data_clean.duration > 0)]
    total_call_dest_nums.append(len(all_call_dest_num))
    total_call_dest_nums_exc.append(len(all_call_dest_num_exc))
    
    # neighbor
    neighbor_source = find_neighbor_day_dest(num[0], num[1], num[2], data_clean.source_num, data_clean.day, data_clean.dest_num, 50)
    neighbor_sources.append(len(neighbor_source))
    neighbor = data_clean[(data_clean.source_num.isin(neighbor_source)) & (data_clean.day == num[1]) & (data_clean.dest_num == num[2])]
    neighbor_destinations.append(neighbor.dest_num.nunique())
    neighbor_durations.append(neighbor.duration.sum())
    # blacklist ratio
    l_bl_inc = len(all_call_source_num[all_call_source_num['dest_country_status'] == 'BLACKLIST'])
    if l_bl_inc == 0:
        ratio_blacklist.append(0)
    else:
        ratio_blacklist.append(l_bl_inc/len(all_call_source_num) * 100)
    l_bl_exc = len(all_call_source_num_exc[all_call_source_num_exc['dest_country_status'] == 'BLACKLIST'])
    if l_bl_exc == 0:
        ratio_blacklist_exc.append(0)
    else:
        ratio_blacklist_exc.append(l_bl_exc/len(all_call_source_num_exc) * 100)
```

```{.json .output n=46}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "time: 998 \u00b5s\n"
 }
]
```

```{.python .input}
data_ready_3['avg_interval_dest_num'] = intervals
data_ready_3['total_duration_source_num'] = total_duration_source_nums
data_ready_3['total_duration_source_num_exc'] = total_duration_source_nums_exc
data_ready_3['total_call_source_num'] = total_call_source_nums
data_ready_3['total_call_source_num_exc'] = total_call_source_nums_exc
data_ready_3['total_call_dest_num'] = total_call_dest_nums
data_ready_3['total_call_dest_num_exc'] = total_call_dest_nums_exc
data_ready_3['avg_interval_all_dest_num'] = intervals_all_dest_num
data_ready_3['avg_interval_dest_num'] = data_ready_3['avg_interval_dest_num'].fillna(0)
data_ready_3['avg_interval_all_dest_num'] = data_ready_3['avg_interval_all_dest_num'].fillna(0)
data_ready_3['ratio_blacklist'] = ratio_blacklist_inc
data_ready_3['ratio_blacklist_exc'] = ratio_blacklist_exc
data_ready_3['nunique_source_num_neighbor'] = neighbor_sources
data_ready_3['nunique_dest_num_neighbor'] = neighbor_destinations
data_ready_3['total_duration_source_num_neighbor'] = neighbor_durations
# data_ready_3.head()
```

```{.python .input}
data_ready_3.to_csv('dataset/fraud_data_ready_day_dest_num.csv', index=True)
```

___

# Modeling
Selanjutnya mulai melakukan tahap modeling dengan menggunakan **Anomaly
Detection** dari `sklearn`, langkah-langkahnya sebagai berikut :

## Load Data Ready

Load kembali file data ready yang ingin dimodelkan, kemudian buang data yang
memiliki `total_duration` nya 0.

```{.python .input  n=7}
# Jika menggunakan Data Ready 1
data_ready_1 = pd.read_csv('dataset/fraud_data_ready.csv', index_col=[0], skipinitialspace=True)

# Jika menggunakan Data Ready 2
data_ready_2 = pd.read_csv('dataset/fraud_data_ready_day.csv', index_col=[0,1], skipinitialspace=True)

# Jika menggunakan Data Ready 3
data_ready_3 = pd.read_csv('dataset/fraud_data_ready_day_dest_num_2.csv', index_col=[0,1,2], skipinitialspace=True)
```

```{.json .output n=7}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "time: 1.62 s\n"
 }
]
```

## Data Scaling

```{.python .input  n=8}
scaler = StandardScaler()

# Jika menggunakan Data Ready 1
data_scale_1 = scaler.fit_transform(data_ready_1)

# Jika menggunakan Data Ready 2
data_scale_2 = scaler.fit_transform(data_ready_2)

# Jika menggunakan Data Ready 3
data_scale_3 = scaler.fit_transform(data_ready_3)
```

```{.json .output n=8}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "time: 265 ms\n"
 }
]
```

```{.python .input  n=9}
# Jika menggunakan Data Ready 1
dfready_scale_1 = data_ready_1.copy()
dfready_scale_1.iloc[:,:] = data_scale_1

# Jika menggunakan Data Ready 2
dfready_scale_2 = data_ready_2.copy()
dfready_scale_2.iloc[:,:] = data_scale_2

# Jika menggunakan Data Ready 3
dfready_scale_3 = data_ready_3.copy()
dfready_scale_3.iloc[:,:] = data_scale_3
```

```{.json .output n=9}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "time: 821 ms\n"
 }
]
```

## Dimensionality Reduction

Menggunakan elbow method seperti di bawah ini untuk menentukan number of
component yang digunakan nantinya pada saat scaling.

```{.python .input  n=104}
rcParams['figure.figsize'] = 30, 5
anomaly_algorithms = [('Data Ready 1', dfready_scale_1), # model one class svm disimpan dengan nama `ocsvm` 
                      ('Data Ready 2',dfready_scale_2), 
                      ('Data Ready 3',dfready_scale_3)]
plot_num = 1
xx, yy = np.meshgrid(np.linspace(-10, 100, 300),np.linspace(-20, 40, 200))
for name, algorithm in anomaly_algorithms:
    t0 = time.time()
    pca = PCA(random_state=1).fit(algorithm)
    t1 = time.time()
    plt.subplot(1, len(anomaly_algorithms), plot_num)
    plt.title(name, size=18)

    colors = np.array(['#377eb8', '#ff7f00'])
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('number of components')
    plt.ylabel('explained variance')

    plt.text(.99, .01, ('%.2fs' % (t1 - t0)).lstrip('0'),
             transform=plt.gca().transAxes, size=15,
             horizontalalignment='right')
    plot_num+=1

plt.show()
```

```{.json .output n=104}
[
 {
  "data": {
   "image/png": "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zpuwdf91iy5HkiRptfXamzN48PHXWb/nGuy7Y9+iy5EkqWF1KrqAos2eO5+Lr5vEX6fOYLeBG3Lk0C2d6JIkSVqGPz79Nx566g36rb82h+6xWdHlSJIkrdauv/s5Pmpu5qt7bU6njh5zLklSvbTrvez8BR9xxR3P8MJf32GXz/ThuC8MoIMTXZKkBvPb397KYYcdyD777MqIEd/htddebbXd669PYd99d+Ott95s4wpVFtNnzOHK256kS6cOfPvLW9O5U7uOkpKkBrSyuQng+98fwfDh327D6lQ2z778NpNfeJMB/T7Fdlv0KrocSZJqatasDxg16hcceuiXGDp0D04++XieeurJZbavd3Zq17+heH3aB7zwynQ+v82GHP+lrejQwYkuSVJjueeeuxgz5lqGDx/BtdfeSNeuXTnjjFOZO3fuYu1effUVTj/9FGbPnl1QpSqDF155h5kfzuPwfbZgo15rFV2OJEk1tbK5CeCuu+5g4sSHC6hSABHRISKujog/RcSEiNh8iZ9/LSL+HBETq+0K+f3X8y9Pp2OHJo7YZwtXEZIkNZwLL7yARx75Ez/84fmMGXMzEVvxve+dwquvvrJU27bITu16sqtfn7X5yfE7c9YxO3kquSSpId16600cfviR7LXXvmy22eace+5PmT59Ovfff/+iNrfd9htOOOEY1l67e4GVqgx23qoPo//3Xuy13cZFlyJJUs0tKzdNmDB+sXZTprzGtddexWc/O7CgSgUMA7pl5ueBHwCXLPxBRKwBXADslZmDgXWAA4so8su79edXZ+1Dvz7mbElSY3n//fd48MHfc+qp32P77XekX79N+e53z6BXr9488MC4xdq2VXZq1zM8TU1NbNx7bTp6RpckqQFNn/4Or732Ktttt8OibWuuuSYDBmzF5MmTF22bNOmPnHXWPzN8+IhW+7nllhv4ylcOZq+9Ps8RRxzKHXfcVvfatXrq1LED/TboUXQZkiTV3PJy09NPP7Fo24IFC7jggnM56qhj6N//fy3Vj7mpzewG3AeQmZOAHVv8bA4wODNnVe93AgpZvqBbl05s6NnwkqQG1LlzFy6++HK23XbQom1NTU00NTUxY8b7i7a1ZXbqtMqPlCRJq7W33noLgN69119se69evZk6deqi+5deeiUAjz8+mSU9/PBD/PrXN/PjH/+Mvn034dFH/8xFF/2UzTbbnEGDtq9j9ZIkSW1nebmp5fVMb77532hqgq997Wguuuini7U1N7WpHsB7Le4viIhOmTk/Mz8C3gSIiFOBtYH/t6IOe/Zck06dOtal2N69G/vMLsdXbo6v3BxfuZV7fN3p12//xbbcf//9TJnyGvvttw9QGd9VV11F584dOe207zBy5Ei6dOm0aNzjx49n7NhbGDVqFP369WPixImMHDmSHXYYyE477fSJK3KyS5KkBrXw+ltdunRZbHvnzp2ZOfO91h6ylNdff43OnTuxwQYbssEGG3LQQcPYaKON2XTT/rUuV5IkqTDLy01z5lSu2ZX5ImPH3sp1191Ehw5LL5RjbmpT7wMtf0PYITPnL7xTvUbXRcCWwGGZ2byiDqdPn7WiJqukd+/uTJs2oy59rw4cX7k5vnJzfOXWaON77rlnOfvssxkyZG8GDKic7fXww48yZsz1XHfdTbz99gfMnj2PuXPnLxr388+/RMeOHenWbR26dOnBkCFfYNSo9VhnnfWX+dosb4KwXS9jKElSI+vatSsA8+bNW2z7vHnzWGONNVaqj/32O4AePdbhiCMO4dhjj2D06Mvp0aMHPXuuW/N6JUmSirL83NSNOXPm8JOfjORb3zqZvn03abUPc1Ob+iPwRYCI+BzwzBI/vwboBgxrsZyhJEmqg0mTJjJixHcYMGBrRo78MUAh2cnJLkmSGlSfPn0AePvtvy+2/e9/n7boZ4/MdZgAAA6FSURBVCvSs+e63HjjWK688loGD96dyZP/zAknHMO4cffVvF5JkqSiLC839eq1Pk899RR//evLXH31FQwdujtDh+7Ovffew9NPP8nQobszdepUc1PbuhOYHRETgcuA0yPiyIj4dkRsDxwPbAOMj4gJEXFIkcVKktSo7r33Hs4663R22mkXLr54FF27dgMoJDu5jKEkSQ2qZ8916du3H0888RjbbrsdALNmzeLFF1/g6KOPWqk+HnhgHO+++y6HHfZVBg3anhNPPIUzzzyNceP+L/vt94V6li9JktRmlpebDj74UAYOHMjYsXcu9phrrhnN1KlvcO65F9CrVy9zUxuqXpfrpCU2v9jitgd3S5JUZw88MI5/+ZfzOfDAYZx55g/o2PHja18WkZ2c7JIkqcHMnDmTefPm0bNnT4444khGj76cvn034dOf3oxrrhnNeuv1YujQobz3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\n",
   "text/plain": "<Figure size 2160x360 with 3 Axes>"
  },
  "metadata": {
   "needs_background": "light"
  },
  "output_type": "display_data"
 },
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "time: 739 ms\n"
 }
]
```

Dari hasil plot elbow method di atas telah didapatkan number of components yang
bisa digunakan, dalam kasus ini digunakan pc=8. Tapi untuk visualisasi nanti
akan dilakukan setelah data prediksi didapatkan dengan pc=2 karna ingin didapat
data 2 dimensi untuk bisa di-visualisasikan.

```{.python .input  n=10}
# pc_1 = 8, pc_2 = 8, pc_3 = 8

pc = 8
pca_scale = PCA(pc, random_state=1)
dfready_scale_pca_1 = pca_scale.fit_transform(dfready_scale_1)
dfready_scale_pca_2 = pca_scale.fit_transform(dfready_scale_2)
dfready_scale_pca_3 = pca_scale.fit_transform(dfready_scale_3)

# dfready_scale_pca.shape
```

```{.json .output n=10}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "time: 1.26 s\n"
 }
]
```

## Anomaly Detection

```{.python .input  n=11}
# Anomaly Detection
ocsvm_1 = svm.OneClassSVM(nu=0.1)
rocov_1 = EllipticEnvelope(random_state=1)
isofor_1 = IsolationForest(random_state=1)

ocsvm_2 = svm.OneClassSVM(nu=0.1)
rocov_2 = EllipticEnvelope(random_state=1)
isofor_2 = IsolationForest(random_state=1)

ocsvm_3 = svm.OneClassSVM(nu=0.1)
rocov_3 = EllipticEnvelope(random_state=1)
isofor_3 = IsolationForest(random_state=1)
```

```{.json .output n=11}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "time: 1.99 ms\n"
 }
]
```

### Model Training
Melakukan training data dengan model yang digunakan.

```{.python .input  n=97}
ocsvm_1.fit(dfready_scale_pca_1)
rocov_1.fit(dfready_scale_pca_1)
isofor_1.fit(dfready_scale_pca_1)

ocsvm_2.fit(dfready_scale_pca_2)
rocov_2.fit(dfready_scale_pca_2)
isofor_2.fit(dfready_scale_pca_2)

ocsvm_3.fit(dfready_scale_pca_3)
rocov_3.fit(dfready_scale_pca_3)
isofor_3.fit(dfready_scale_pca_3)
```

```{.json .output n=97}
[
 {
  "data": {
   "text/plain": "IsolationForest(random_state=1)"
  },
  "execution_count": 97,
  "metadata": {},
  "output_type": "execute_result"
 },
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "time: 23min 39s\n"
 }
]
```

Simpan ke dalam file untuk kebutuhan analisa lanjutan

```{.python .input  n=98}
dump(ocsvm_1, 'dataset/ocsvm_1.joblib')
dump(rocov_1, 'dataset/rocov_1.joblib')
dump(isofor_1, 'dataset/isofor_1.joblib')

dump(ocsvm_2, 'dataset/ocsvm_2.joblib')
dump(rocov_2, 'dataset/rocov_2.joblib')
dump(isofor_2, 'dataset/isofor_2.joblib')

dump(ocsvm_3, 'dataset/ocsvm_3.joblib')
dump(rocov_3, 'dataset/rocov_3.joblib')
dump(isofor_3, 'dataset/isofor_3.joblib')
```

```{.json .output n=98}
[
 {
  "data": {
   "text/plain": "['dataset/isofor_3.joblib']"
  },
  "execution_count": 98,
  "metadata": {},
  "output_type": "execute_result"
 },
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "time: 680 ms\n"
 }
]
```

### Model Test

Fungsi decision_function untuk melihat score secara real, jika nilai score
anomali semakin minus maka semakin anomali.

```{.python .input  n=81}
ocsvm_1 = load('dataset/ocsvm_1.joblib')
rocov_1 = load('dataset/rocov_1.joblib')
isofor_1 = load('dataset/isofor_1.joblib')

ocsvm_2 = load('dataset/ocsvm_2.joblib')
rocov_2 = load('dataset/rocov_2.joblib')
isofor_2 = load('dataset/isofor_2.joblib')

ocsvm_3 = load('dataset/ocsvm_3.joblib')
rocov_3 = load('dataset/rocov_3.joblib')
isofor_3 = load('dataset/isofor_3.joblib')
```

```{.json .output n=81}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "time: 31.3 ms\n"
 }
]
```

```{.python .input  n=99}
score_rocov_1 = rocov_1.decision_function(dfready_scale_pca_1)
score_ocsvm_1 = ocsvm_1.decision_function(dfready_scale_pca_1)
score_isofor_1 = isofor_1.decision_function(dfready_scale_pca_1)

score_rocov_2 = rocov_2.decision_function(dfready_scale_pca_2)
score_ocsvm_2 = ocsvm_2.decision_function(dfready_scale_pca_2)
score_isofor_2 = isofor_2.decision_function(dfready_scale_pca_2)

score_rocov_3 = rocov_3.decision_function(dfready_scale_pca_3)
score_ocsvm_3 = ocsvm_3.decision_function(dfready_scale_pca_3)
score_isofor_3 = isofor_3.decision_function(dfready_scale_pca_3)
```

```{.json .output n=99}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "time: 2min 55s\n"
 }
]
```

```{.python .input  n=100}
# Membuat prediksi dari masing-masing model
pred_rocov_1 = rocov_1.predict(dfready_scale_pca_1)
pred_ocsvm_1 = ocsvm_1.predict(dfready_scale_pca_1)
pred_isofor_1 = isofor_1.predict(dfready_scale_pca_1)

pred_rocov_2 = rocov_2.predict(dfready_scale_pca_2)
pred_ocsvm_2 = ocsvm_2.predict(dfready_scale_pca_2)
pred_isofor_2 = isofor_2.predict(dfready_scale_pca_2)

pred_rocov_3 = rocov_3.predict(dfready_scale_pca_3)
pred_ocsvm_3 = ocsvm_3.predict(dfready_scale_pca_3)
pred_isofor_3 = isofor_3.predict(dfready_scale_pca_3)
```

```{.json .output n=100}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "time: 2min 53s\n"
 }
]
```

```{.python .input  n=101}
# Simpan index dimana anomali data ditangkap
idx_rocov_1 = np.where(pred_rocov_1 == -1)
idx_ocsvm_1 = np.where(pred_ocsvm_1 == -1)
idx_isofor_1 = np.where(pred_isofor_1 == -1)

idx_rocov_2 = np.where(pred_rocov_2 == -1)
idx_ocsvm_2 = np.where(pred_ocsvm_2 == -1)
idx_isofor_2 = np.where(pred_isofor_2 == -1)

idx_rocov_3 = np.where(pred_rocov_3 == -1)
idx_ocsvm_3 = np.where(pred_ocsvm_3 == -1)
idx_isofor_3 = np.where(pred_isofor_3 == -1)
```

```{.json .output n=101}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "time: 4.95 ms\n"
 }
]
```

```{.python .input  n=85}
np.where(pred_rocov_1 == -1)
np.where(pred_ocsvm_1 == -1)
np.where(pred_isofor_1 == -1)
```

```{.json .output n=85}
[
 {
  "data": {
   "text/plain": "(array([    2,     7,    11, ..., 44099, 44100, 44101], dtype=int64),)"
  },
  "execution_count": 85,
  "metadata": {},
  "output_type": "execute_result"
 },
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "time: 1.96 ms\n"
 }
]
```

### Prediction

Prediksi dan dapatkan kira-kira data apa saja yang terindikasi fraudulent.
Berikut adalah gambaran hasil yang diberikan (dari data ready)

```{.python .input  n=108}
data_predict_svm_1 = data_ready_1.iloc[idx_ocsvm_1[0],:]
data_predict_svm_2 = data_ready_2.iloc[idx_ocsvm_2[0],:]
data_predict_svm_3 = data_ready_3.iloc[idx_ocsvm_3[0],:]

dump(data_predict_svm_1, 'dataset/data_predict_svm_1.joblib')
dump(data_predict_svm_2, 'dataset/data_predict_svm_2.joblib')
dump(data_predict_svm_3, 'dataset/data_predict_svm_3.joblib')
```

```{.json .output n=108}
[
 {
  "data": {
   "text/plain": "['dataset/data_predict_svm_3.joblib']"
  },
  "execution_count": 108,
  "metadata": {},
  "output_type": "execute_result"
 },
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "time: 75.8 ms\n"
 }
]
```

```{.python .input  n=109}
data_predict_cov_1 = data_ready_1.iloc[idx_rocov_1[0],:]
data_predict_cov_2 = data_ready_2.iloc[idx_rocov_2[0],:]
data_predict_cov_3 = data_ready_3.iloc[idx_rocov_3[0],:]

dump(data_predict_cov_1, 'dataset/data_predict_cov_1.joblib')
dump(data_predict_cov_2, 'dataset/data_predict_cov_2.joblib')
dump(data_predict_cov_3, 'dataset/data_predict_cov_3.joblib')
```

```{.json .output n=109}
[
 {
  "data": {
   "text/plain": "['dataset/data_predict_cov_3.joblib']"
  },
  "execution_count": 109,
  "metadata": {},
  "output_type": "execute_result"
 },
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "time: 52.1 ms\n"
 }
]
```

```{.python .input  n=110}
data_predict_isofor_1 = data_ready_1.iloc[idx_isofor_1[0],:]
data_predict_isofor_2 = data_ready_2.iloc[idx_isofor_2[0],:]
data_predict_isofor_3 = data_ready_3.iloc[idx_isofor_3[0],:]

dump(data_predict_isofor_1, 'dataset/data_predict_isofor_1.joblib')
dump(data_predict_isofor_2, 'dataset/data_predict_isofor_2.joblib')
dump(data_predict_isofor_3, 'dataset/data_predict_isofor_3.joblib')
```

```{.json .output n=110}
[
 {
  "data": {
   "text/plain": "['dataset/data_predict_isofor_3.joblib']"
  },
  "execution_count": 110,
  "metadata": {},
  "output_type": "execute_result"
 },
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "time: 46.9 ms\n"
 }
]
```

### 2d Visualization
Lalu interpretasikan (dan cocokkan) data-data yang anomali dari hasil reduksi
dimensi menggunakan visualisasi plotly berikut :

```{.python .input  n=38}
df1 = pd.DataFrame(dfready_scale_pca_1,columns=['pc1','pc2','pc3','pc4','pc5','pc6','pc7','pc8'],index=data_ready_1.index)
df2 = pd.DataFrame(dfready_scale_pca_2,columns=['pc1','pc2','pc3','pc4','pc5','pc6','pc7','pc8'],index=data_ready_2.index)
df3 = pd.DataFrame(dfready_scale_pca_3,columns=['pc1','pc2','pc3','pc4','pc5','pc6','pc7','pc8'],index=data_ready_3.index)
```

```{.json .output n=38}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "time: 17.9 ms\n"
 }
]
```

```{.python .input  n=39}
pc = 2
pca_scale = PCA(pc, random_state=1)
dfready_pca_1 = pca_scale.fit_transform(df1)
dfready_pca_2 = pca_scale.fit_transform(df2)
dfready_pca_3 = pca_scale.fit_transform(df3)

# dfready_tsne_1 = TSNE(n_components=pc).fit_transform(df1)
# dfready_tsne_2 = TSNE(n_components=pc).fit_transform(df2)
# dfready_tsne_3 = TSNE(n_components=pc).fit_transform(df3)

# dfready_tsne_1 = TSNE(n_components=pc).fit_transform(dfready_scale_1)
# dfready_tsne_2 = TSNE(n_components=pc).fit_transform(dfready_scale_2)
# dfready_tsne_3 = TSNE(n_components=pc).fit_transform(dfready_scale_3)
```

```{.json .output n=39}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "time: 2.82 s\n"
 }
]
```

```{.python .input  n=40}
df_tsne_1 = pd.DataFrame(dfready_pca_1,columns=['pc1','pc2'],index=data_ready_1.index)
df_tsne_2 = pd.DataFrame(dfready_pca_2,columns=['pc1','pc2'],index=data_ready_2.index)
df_tsne_3 = pd.DataFrame(dfready_pca_3,columns=['pc1','pc2'],index=data_ready_3.index)

# df_tsne_1 = pd.DataFrame(dfready_tsne_1,columns=['pc1','pc2'],index=data_ready_1.index)
# df_tsne_2 = pd.DataFrame(dfready_tsne_2,columns=['pc1','pc2'],index=data_ready_2.index)
# df_tsne_3 = pd.DataFrame(dfready_tsne_3,columns=['pc1','pc2'],index=data_ready_3.index)
```

```{.json .output n=40}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "time: 1.99 ms\n"
 }
]
```

```{.python .input  n=41}
# Join data hasil scaling di atas dengan data_ready yang di atas.
df_1 = df_tsne_1.join(data_ready_1)
df_2 = df_tsne_2.join(data_ready_2)
df_3 = df_tsne_3.join(data_ready_3)
```

```{.json .output n=41}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "time: 49.9 ms\n"
 }
]
```

```{.python .input  n=42}
df_pca_1 = df_tsne_1.copy()
df_pca_2 = df_tsne_2.copy()
df_pca_3 = df_tsne_3.copy()
```

```{.json .output n=42}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "time: 11 ms\n"
 }
]
```

```{.python .input  n=44}
# Menggunakan Data Ready 1
df = df_1.copy()
fig = px.scatter(df, x="pc1", y="pc2", hover_data=[df.index.get_level_values(0), 
    df.index, 
    'pc1','pc2', 'total_call', 'total_duration'])
# fig.show()
plotly.offline.plot(fig, filename='assets/plt_data_ready_1.html', image='png', auto_open=True, output_type='file', image_width=800, image_height=600, validate=False)
```

```{.json .output n=44}
[
 {
  "data": {
   "text/plain": "'assets/plt_data_ready_1.html'"
  },
  "execution_count": 44,
  "metadata": {},
  "output_type": "execute_result"
 },
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "time: 1.6 s\n"
 }
]
```

Menggunakan data ready 1 maka akan menghasilkan plot seperti berikut :

![ ](assets/plt_data_ready_1.png)

```{.python .input  n=45}
#Menggunakan Data Ready 2
df = df_2.copy()
fig = px.scatter(df, x="pc1", y="pc2", hover_data=[df.index.get_level_values(0), 
    df.index.get_level_values(1), 
    'pc1','pc2', 'total_call', 'total_duration'])
# fig.show()
plotly.offline.plot(fig, filename='assets/plt_data_ready_2.html', image='png', auto_open=True, output_type='file', image_width=800, image_height=600, validate=False)
```

```{.json .output n=45}
[
 {
  "data": {
   "text/plain": "'assets/plt_data_ready_2.html'"
  },
  "execution_count": 45,
  "metadata": {},
  "output_type": "execute_result"
 },
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "time: 6.14 s\n"
 }
]
```

Menggunakan data ready 2 maka akan menghasilkan plot seperti berikut :

![ ](assets/plt_data_ready_2.png)

```{.python .input  n=46}
#Menggunakan Data Ready 3
df = df_3.copy()
fig = px.scatter(df, x="pc1", y="pc2", hover_data=[df.index.get_level_values(0), 
    df.index.get_level_values(1), 
    df.index.get_level_values(2),
    'pc1','pc2', 'total_call', 'total_duration'])
# fig.show()
plotly.offline.plot(fig, filename='assets/plt_data_ready_3.html', image='png', auto_open=True, output_type='file', image_width=800, image_height=600, validate=False)
```

```{.json .output n=46}
[
 {
  "data": {
   "text/plain": "'assets/plt_data_ready_3.html'"
  },
  "execution_count": 46,
  "metadata": {},
  "output_type": "execute_result"
 },
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "time: 10.8 s\n"
 }
]
```

Menggunakan data ready 3 maka akan menghasilkan plot seperti berikut :

![ ](assets/plt_data_ready_3.png)

Performa masing-masing model **Anomaly Detection** seperti gambar berikut :

```{.python .input  n=47}
# Dengan menggunakan data ready 1
rcParams['figure.figsize'] = 30, 5
anomaly_algorithms = [('One Class SVM', ocsvm_1), # model one class svm disimpan dengan nama `ocsvm` 
                      ('Robust Covariance',rocov_1), 
                      ('Isolation Forest',isofor_1)]
plot_num = 1
xx, yy = np.meshgrid(np.linspace(-10, 100, 300),np.linspace(-20, 40, 200))
for name, algorithm in anomaly_algorithms:
    model = algorithm
    t0 = time.time()
    model.fit(df_pca_1)
    t1 = time.time()
    plt.subplot(1, len(anomaly_algorithms), plot_num)

    plt.title(name, size=18)

    # fit the data and tag outliers
    y_pred = algorithm.predict(df_pca_1)

    # plot the levels lines and the points
    Z = algorithm.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors='black')

    colors = np.array(['#377eb8', '#ff7f00'])
    plt.scatter(df_pca_1['pc1'], df_pca_1['pc2'], s=10, color=colors[(y_pred + 1) // 2])
    plt.xlim(-20, 100)
    plt.ylim(-20, 50)
    plt.xticks(())
    plt.yticks(())
    plt.text(.99, .01, ('%.2fs' % (t1 - t0)).lstrip('0'),
             transform=plt.gca().transAxes, size=15,
             horizontalalignment='right')
    plot_num+=1

plt.show()
```

```{.json .output n=47}
[
 {
  "data": {
   "image/png": "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AAAAAqHABAAAAAAAAAAAgEAoMAEAAAAAAAAAACAQCkwAAAAAAAAAAAAIhAITAAAAAAAAAAAAAqHABAAAAAAAAAAAgEAoMAEAAAAAAAAAACAQCkwAAAAAAAAAAAAIhAITAAAAAAAAAAAAAqHABAAAAAAAAAAAgEAoMAEAAAAAAAAAACAQCkwAAAAAAAAAAAAIhAITAAAAAAAAAAAAAqHABAAAAAAAAAAAgEAoMAEAAAAAAAAAACAQCkwAAAAAAAAAAAAIhAITAAAAAAAAAAAAAqHABAAAAAAAAAAAgEAoMAEAAAAAAAAAACAQCkwAAAAAAAAAAAAIhAITAAAAAAAAAAAAAqHABAAAAAAAAAAAgEAoMAEAAAAAAAAAACAQCkwAAAAAAAAAAAAIhAITAAAAAAAAAAAAAqHABAAAAAAAAAAAgEAoMAEAAAAAAAAAACAQCkwAAAAAAAAAAAAIhAITAAAAAAAAAAAAAqHABAAAAAAAAAAAgEAoMAEAAAAAAAAAACAQCkwAAAAAAAAAAAAIhAITAAAAAAAAAAAAAqHABAAAAAAAAAAAgEAoMAEAAAAAAAAAACAQCkwAAAAAAAAAAAAIhAITAAAAAAAAAAAAAqHABAAAAAAAAAAAgEAoMAEAAAAAAAAAACAQCkwAAAAAAAAAAAAIhAITAAAAAAAAAAAAAqHABAAAAAAAAAAAgEAoMAEAAAAAAAAAACAQCkwAAAAAAAAAAAAIhAITAAAAAAAAAAAAAqHABAAAAAAAAAAAgEAoMAEAAAAAAAAAACAQCkwAAAAAAAAAAAAIhAITAAAAAAAAAAAAAqHABAAAAAAAAAAAgEAoMAEAAAAAAAAAACAQCkwAAAAAAAAAAAAIhAITAAAAAAAAAAAAAqHABAAAAAAAAAAAgEAoMAEAAAAAAAAAACAQCkwAAAAAAAAAAAAIhAITAAAAAAAAAAAAAqHABAAAAAAAAAAAgEAoMAEAAAAAAAAAACAQCkwAAAAAAAAAAAAIhAITAAAAAAAAAAAAAqHABAAAAAAAAAAAgEAoMAEAAAAAAAAAACAQCkwAAAAAAAAAAAAIhAITAAAAAAAAAAAAAqHABAAAAAAAAAAAgEAoMAEAAAAAAAAAACAQCkwAAAAAAAAAAAAIhAITAAAAAAAAAAAAAqHABAAAAAAAAAAAgEAoMAEAAAAAAAAAACAQCkwA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9fzhD/NZv37tScVkOfmXIyIi0jX0738a3bv3YP78p5gzZy6hoSG89tqfKCkpxul0HnV9ZWUlH320hDFjxvH44/PIz8/j+efnU1JSzKOP/vao6z/44F3ee+9t7rvvl4SHR1BWVkppaQlRUdEkJCSSmJjEk08+0+jPEhEREfE38+f/jvHjz2TMmHHk5eUe99qKigpycrK59trrOffc89m9excvvPAcixYt5Oabbz/qeuVNIiIiIj6bN2/iqafmcvbZ5zB69NgG9yUmJvHnP/+NHTu2s2DBs0RERHL77XeRlXUAq9VCQkIiCQmJXHzxdJKSkunevcdJxaACk4iIyAlYrVaefPIZ5s6dw/TpF2C1Wjn//AsYO3Y8Fov1qOstFgthYWHMmTMXs9lMv34DcLlczJnzELNn30d4eET9ta+//hoLF77CrFk3cvnlVwEQFhbO1Vdfx3PP/Z6//OVPjBgxmvHjz+Scc85vs9csIiIicjI++eQjduzYweuvv9Wk61955QVMJjN33jkbgL59++F2u3nmmae44oqrlDeJiIiINOK771YwZ85DDBgwkDlz5h51f3h4BOHhEfTpk05xcRGLFi3kllvuYNKkC/nooyVcffWl9OrVm1GjxjJp0gVERkadVBwqMImIiDRB9+49eO21NygtLcFqtRIUFMxNN808qmULQExMHHa7DbPZXH9bjx49AcjOziY8PAKPx8Ozz85jyZL3ufPO2cyceUOD57jrrnu47LIZrFjxLT/88B1PPTWXzz77hPnzF7TuCxURERE5BR9//C/y83OZNm0yAF6vF4AHHriHCy+cyi9/+XCD6zdv3sSECWc3uG3AgIG4XC5yc3OUN4mIiIj8xCeffMS82y6RAAAABBZJREFUeY8zfvwEHnvsd9hstvr71q1bQ0hICH36pNff1qtXb2praykrKyMyMorXX/8HGzas5/vvV/Ldd//h7bcX88gjv2XSpAuaHYtmMImIiJxAVVUld999G3v27CI8PIKgoGCysw+ya9dORo06usA0ePAQdu7cgcvlqr9tz57dmM1mEhMTAXjuuaf56KMlPPzwb45aJMnKyuSZZ54iKiqayy+/knnznuORR37LypX/obi4qHVfrIiIiMgp+PWvH+fNN99h0aLFLFq0mGeffQmAhx56lFtuueOo6+Pi4ti9e2eD2/bu3Y3JZCI5uRugvElERETkkGXLPuPJJ3/LlCmX8Pjj8xoUlwD+/vfXWbjwlQa3bdmymcjIKCIiIli27DM++OBdhgwZxu2338WiRYsZOXI0n3328UnFoxNMIiIiJxAUFIzb7WbBgue4994HqK6u4qmn5jJ8+EiGDx+J0+mkrKyUsLBwrFYr06dfznvv/ZMnnvgNN954K3l5ufzxjwuYPHkK4eERrFz5LR9++C433ngro0ePpbCwoP5nhYSEEh4ewbJlS3E4HMyceQNer5cvvviM5ORuDdrEiIiIiPib2Ni4Bt8fWvSIiYklMjLqqLxpxoyrefDB+/jrX//M+edfwL59e3nxxee59NIrCA4OUd4kIiIiXV5FRQVOpxOv18O8eU8wcuRobrnldkpKiuuvCQgIIDg4hKuuupb775/N4sV/Y8KEiaxfv4bFi//G7Nn3YRgGDoeDl19eQGhoKIMGDSEz8wA7dmxn+vTLTyo249Bx9cbk55cf+04REZEuJC8vl+eff5q1a1djtwdw1lnncOedswkKCmLt2tX8/Od38MILrzJs2AgA9u7dw4svPsePP64jMDCIyZMv5Pbb78Zms/HYY4/w+ef/bvTnzJkzl8mTp7B58yZeeeUFdu7cjsfjZciQocyefT+pqd0bfVxsbKjRai9emky5k4iISEN5eblcdtnU+jypsbzp66+/4vXXXyMjYx9RUdFccMFUZs26EYvForypk1PuJCIicmK/+91jrFu3hlmzbmT+/Ccbveaii6bx0ENzAFi+/Av+8peFHDiQQVxcPNdddz0XXTS9/trFi99gyZL3yMvLJTIyikmTLuSWW+7AYmn8PNLxcicVmERERDoBLZT4B+VOIiIi/k95k/9Q7iQiIuL/jpc7aQaTiIiIiIiIiIiIiIiINIsKTCIiIiIiIiIiIiIiItIsx22RJyIiIiIiIiIiIiIiIvJTOsEkIiIiIiIiIiIiIiIizaICk4iIiIiIiIiIiIiIiDSLCkwiIiIiIiIiIiIiIiLSLCowiYiIiIiIiIiIiIiISLOowCQiIiIiIiIiIiIiIiLNogKTiIiIiIiIiIiIiIiINMv/A2fgXaH4yNkWAAAAAElFTkSuQmCC\n",
   "text/plain": "<Figure size 2160x360 with 3 Axes>"
  },
  "metadata": {},
  "output_type": "display_data"
 },
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "time: 28.3 s\n"
 }
]
```

```{.python .input  n=48}
# Dengan menggunakan data ready 2
rcParams['figure.figsize'] = 30, 5
anomaly_algorithms = [('One Class SVM', ocsvm_2), # model one class svm disimpan dengan nama `ocsvm` 
                      ('Robust Covariance',rocov_2), 
                      ('Isolation Forest',isofor_2)]
plot_num = 1
xx, yy = np.meshgrid(np.linspace(-10, 100, 300),np.linspace(-20, 40, 200))
for name, algorithm in anomaly_algorithms:
    model = algorithm
    t0 = time.time()
    model.fit(df_pca_2)
    t1 = time.time()
    plt.subplot(1, len(anomaly_algorithms), plot_num)

    plt.title(name, size=18)

    # fit the data and tag outliers
    y_pred = algorithm.predict(df_pca_2)

    # plot the levels lines and the points
    Z = algorithm.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors='black')

    colors = np.array(['#377eb8', '#ff7f00'])
    plt.scatter(df_pca_2['pc1'], df_pca_2['pc2'], s=10, color=colors[(y_pred + 1) // 2])
    plt.xlim(-20, 100)
    plt.ylim(-20, 50)
    plt.xticks(())
    plt.yticks(())
    plt.text(.99, .01, ('%.2fs' % (t1 - t0)).lstrip('0'),
             transform=plt.gca().transAxes, size=15,
             horizontalalignment='right')
    plot_num+=1

plt.show()
```

```{.json .output n=48}
[
 {
  "data": {
   "image/png": "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\n",
   "text/plain": "<Figure size 2160x360 with 3 Axes>"
  },
  "metadata": {},
  "output_type": "display_data"
 },
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "time: 5min 20s\n"
 }
]
```

```{.python .input  n=49}
# Dengan menggunakan data ready 3
rcParams['figure.figsize'] = 30, 5
anomaly_algorithms = [('One Class SVM', ocsvm_3), # model one class svm disimpan dengan nama `ocsvm` 
                      ('Robust Covariance',rocov_3), 
                      ('Isolation Forest',isofor_3)]
plot_num = 1
xx, yy = np.meshgrid(np.linspace(-10, 100, 300),np.linspace(-20, 40, 200))
for name, algorithm in anomaly_algorithms:
    model = algorithm
    t0 = time.time()
    model.fit(df_pca_3)
    t1 = time.time()
    plt.subplot(1, len(anomaly_algorithms), plot_num)

    plt.title(name, size=18)

    # fit the data and tag outliers
    y_pred = algorithm.predict(df_pca_3)

    # plot the levels lines and the points
    Z = algorithm.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors='black')

    colors = np.array(['#377eb8', '#ff7f00'])
    plt.scatter(df_pca_3['pc1'], df_pca_3['pc2'], s=10, color=colors[(y_pred + 1) // 2])
    plt.xlim(-20, 100)
    plt.ylim(-20, 50)
    plt.xticks(())
    plt.yticks(())
    plt.text(.99, .01, ('%.2fs' % (t1 - t0)).lstrip('0'),
             transform=plt.gca().transAxes, size=15,
             horizontalalignment='right')
    plot_num+=1

plt.show()
```

```{.json .output n=49}
[
 {
  "data": {
   "image/png": "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RERERESGTiuYRA4x5xw+hqqmTqqaOlg8p1Srl0REREQGcOVxE2jvDtPZHWbJggmJDkdEREQkqV29aBLpwRTSUlO4cuHERIcjIgkWGGi1Q01NsybxiYiIHARKSvJUTU4Cyp1ERESSn/Km5KHcSUREJPkNlDupRZ6IiIiIiIiIiIiIiIjERQUmERERERERERERERERiYsKTCIiIiIiIiIiIiIiIhIXFZhEREREREREREREREQkLiowiYiIiIiIiIiIiIiISFxUYBIREREREREREREREZG4qMAkIiIiIiIiIiIiIiIicVGBSUREREREREREREREROKiApOIiIiIiIiIiIiIiIjERQUmERERERERERERERERiYsKTCIiIiIiIiIiIiIiIhIXFZhEREREREREREREREQkLiowiYiIiIiIiIiIiIiISFxUYBIREREREREREREREZG4qMAkIiIiIiIiIiIiIiIicVGBSUREREREREREREREROISTHQAIgebiOfx19cqqGvt5PwjxjBpZE6iQxIRERFJWh3dYf70chnhiMcHj51IQXZaokMSERERSVpVTR38fVkFeRlpXLFwAsFUrQ8QkeSlApNInP65Yjv3vF4OQHl9O7d84PAERyQiIiKSvH73ny08tqoKgNauMF9aPDPBEYmIiIgkr188u4nlWxsASEmBDx03KcERiYj0TyVwkTi1dIR6Pu/oDicwEhEREZHkF5svKXcSERERGVhnTL7U1qXcSUSSm1YwicTp4vnj2d7YQWNbN++bNybR4YiIiIgktSULxtPSGSLswZIFExIdjoiIiEhSW7JgAvctqyA7Pcilx4xPdDgiIgMKeJ7X75M1Nc39PykiIiJJo6QkL5DoGES5k4iIyMFAeVPyUO4kIiKS/AbKndQiT0REREREREREREREROKiApOIiIiIiIiIiIiIiIjERQUmERERERERERERERERiYsKTCIiIiIiIiIiIiIiIhIXFZhEREREREREREREREQkLiowiYiIiIiIiIiIiIiISFxUYBIREREREREREREREZG4qMAksh9FPI9wxEt0GCIiIiIHhXDEw/OUO4mIiIgMRSgcSXQIIjLMBRMdgMih6tVNdfz+xTI84OrjJ7FoenGiQxIRERFJWv9csY0HV+wgLzPItafNYGpJTqJDEhEREUlav35uEy9urGNcYRZfO8eQm5mW6JBEZBjSCiaR/eSptTVsb+xgR2MHT9vqRIcjIiIiktSesbXUtnSxubaNJ9dUJTocERERkaTV0R3m+fW11Ld18872Jp5co3EnEUkMFZhE9pOxhVkE/M/HFGQlNBYRERGRZDc6PxOAtNQAk4qzExyNiIiISPJKD6YwusDlTvmZQUxpXoIjEpHhKjBQj/OammY1QBd5lzzP4/E1VRCBM+aUkhIIDP4ikYNAR3eYjGAKAf2dTiolJXn6gSQB5U4i715XKMIjb1cyKj+DRdPUWlgODZ7n0RmKkJmWmuhQJIbypuSh3Enk3atv7eKptdXMGp3H3HEFiQ5HZJ8IRzxCkQgZQeVOyWSg3EkFJhERGZKI53Hr0nWsrGhkakkON547m/SgFsImCw2UJAflTiIiEtXaGeLmf62lvL6NBZOL+Pxp0zVBJ0kob0oeyp1ERCSqrK6VHz6+nsa2Ls49YgxLFkxIdEjiGyh30sigiIgMyY6Gdl7aWEdTR4gV5Y28vmVnokMSERERSVrPrath1Y4mmjpCvLRpJ00doUSHJCIiIpK0nlpbw5a6NurbQ7ywvjbR4cgQqcAkIiJDUpidTl5mEIAR2WnMGJWb4IhEREREktfk4mwy/NXeY/IzyM0IJjgiERERkeRVkpNOqr9OZoy/x5gkPxWYRERkSJauqqKx3c28LcpOY1S+3uxFRERE+vP02ho6QxEARudnkpqirmwiIiIi/Xm9rJ6w3zh14oicxAYjQ6YCk4iIDElzZ2xbFw2QiIiIiAykvTvS87k2mREREREZWEd3eNfnofAAR0oy0Rp9kXdh1fYmnl5bzaTibM6fNzbR4YgcEBcfPZYdDe00toe44MgxiQ5HREQOIi+sr2FFeSNHTSzkxOkjEx2OyAFx2YLxtHaGiABLjhmf6HBEROQg8uCK7ZTvbOP02aOYPSY/0eGIHBCXLZjAfcu2kZMZ5JL54xIdjgxRwPP6n0tVU9OsiVYivYQjHp+9ZwXl9e0EU+C6xTM1UCIiCVdSkqdlZUlAuZPInrbXt3HdfW/T0hkmLyOV/7tsHqPy1GZVRBJHeVPyUO4ksqfn1tXw4yfWE/Zg4ogsbr/8SAIB/bclIokzUO6kFnkicYp4Hu1dbplmKAKNbd0JjkhEREQkeTV3hntyp/buMK2danchIiIi0p/G9lDPPjTt3WEiKsOKSBJTizyROKWlpnDpMeN5dl0NYwuzOGvu6ESHJCIiIpK0zOg8LjxqHKt3NHH42HymjNSGvSIiIiL9OWduKRurW6hs6uBUU0JqilYviUjyUos8ERGRQ4BavSQH5U4iIiLJT3lT8lDuJCIikvzUIk9ERERERERERERERET2GRWYJOl1hSLc89pW/r6sgrAaz4qIiIgMaGdrF3e+VMbjqyoZqFuBiIiIiMDGmhb+8OIWXtu8M9GhiIgcdLQHkyS9Xz63iSfXVAPQ1N7NNSdOSXBEIiIiIsnrtifWsbKiiVS/icHiOdovUkRERKQvXaEIP1y6jsG1r08AACAASURBVIqGDh5fXcVN581m1pj8RIclInLQ0AomSXqrtzf1fL6ztSuBkYiIiIgkt1A4wqaaVgDCHmxv7ExwRCIiIiLJq6qpgx2NHQC0dIYpr29PcEQiIgcXFZgkqb26qY7KJvdGn5oCZxw2KsERiYiIiCSvB97cRktnGICstBQumDcmwRGJiIiIJK8H3txO2O8oXJyTxqmmJLEBiYgcZFRgkqQWjtk2ID8zjRmj8hIXjIiIiEiSC0d2fT6uKJuinPTEBSMiIiKS5GL3qzSleQRTNVQqIhIP7cEkSe34qSO4ZP54Nte2ctzUEeRk6K+siIiISH8unj+OutYuGtq6eN8RWr0kIiIiMpAPHTeR7nCEiAdXHjch0eGIiBx0ArGV+t5qapr7f1JERESSRklJXiDRMYhyJxERkYOB8qbkodxJREQk+Q2UO2ndp4iIiIiIiIiIiIiIiMRFBSaRfWB7Qzuvbt5JOKLJVyIiIiKDWV/dzIqtDQzUTUFERERE3D5RK7Y2sKG6JdGhiIjsQRvaiOylVdsa+cHSddS3dXPCtBF89exZiQ5JREREJGk9vqqKO17YTFc4woVHjePqRZMSHZKIiIhI0rrzpTIeXLGd9GAKHz9xCmfMKU10SCIiPbSCSWQvrahopL6tG4AN1a0JjkZEREQkua3a0UR7KELYg3VVzYkOR0RERCSpratuIeJBR3eEVTuaEh2OiMhuVGAS2UvHTy1mdEEGqQE4YkJBosMRERERSWoLJheRnxkkI5jC0RMKEx2OiIiISFI7ekIhGcEUCrKCLJg8ItHhiIjsJjBQ3/OammY1RZf96q2KBl5YV8uM0lxOMSU8+k4V4woyWTDl4HrDbOsKUd/axdjCLAKBQKLDEZFhqKQkT//5JAHlTrK/PWOrWb29iQWTRzB9VC7P2Brmjs3HjM5LdGhxaWjroisUYVR+ZqJDEZFhSHlT8lDuJPuT53k8uGI72xs6OHtuKRDgzfIGTphWzOiCgysHqW7qICMtlYKstESHIiLD0EC5kwpMkjBdoQifuWcFOxo7SA8GmDoyh7WVLWQEA3zm1OmcYkoSHaKIyEFDAyXJQbmT7E8ba1q44R+raOsKU5gVpCgnnc21bRRmpXHT+2YzfVRuokMUETkoKG9KHsqdZH96YnUVtz+zkYgHE4syaeuOUNvSxYSiLG675Aiy0lMTHaKIyEFhoNxJLfIkYUKRCG1dIQC6Qh41zV0AdIY8NtVqLyMRERGRWA1t3bR3hQFo6wxR2djpHm/vxlZqLyMRERGRWPVt3UT8EmZzR5jaFjfutK2hnZ2tnQmMTETk0BFMdAAyfGWnB3n/UeN4edNOJo7IojQ/g3+vrKQgO53TZmn1koiIiEisoycWct4Ro1lf3cK88QXUt3Xz6uZ6JozI0spvERERkV4uOHIMW+paqW3u4r2zSnhl00421LQyb3wBYwqzEh2eiMghQS3yJKl0hyMEUwLax0hEJE5q9ZIclDvJgabcSUQkfsqbkodyJznQusMR0lLV0ElEJB4D5U5awSRJRW/yIiIiIkOn3ElERERk6JQ7iYjsW/pfVUREREREREREREREROKiApOIiIiIiIiIiIiIiIjERQUmERERERERERERERERiYsKTCIih7iVFY0sL6tPdBgiIiIiSS/ieby8sY51Vc2JDkVEREQk6XWFIjxra9jR2J7oUCRBgokOQERE9p8Hlm/jz69uJRLxuPSY8Xxw4cREhyQiIiKStH757EYeW1VNdloqnzplKiebkkSHJCIiIpKUPM/j5n+vYUV5I8U56dx43iymleQmOiw5wLSCSUTkELauqpnusEfYg/XVLYkOR0RERCSpbaxpBaCtO8yaSq1iEhEREelPKOKxubYNgLrWLlaUNyY4IkkEFZhERA5hx04ZQX5mkNyMVBZMLkp0OCIiIiJJbf6kIjKDKYzMTee4KSMSHY6IiIhI0kpLTeGoCQUEU2DSiCxOnF6c6JAkAQKe5/X7ZE1Nc/9PiojIQaGxvRvP8yjMTk90KLIflZTkBRIdgyh3EhE5FNS1dJKZlkpOhjrKH6qUNyUP5U4iIgc3z/Ooae4kPyuNzLTURIcj+8lAuZMyZhGRQ1xBVlqiQxARERE5aBTnZiQ6BBEREZGDQiAQYFR+ZqLDkARSizwRERERERERERERERGJiwpMIiIiIiIiIiIiIiIiEhe1yBtGOkNhbnxwFU0dIU6aMZIrFk5MdEgiIiIiSauqqYNv/2sNHnDRUWM5bXZpokMSERERSVpvbN7JHS9uITsthatPmMwR4wsTHZKIiOxnWsE0jHz9H6tYU9nCtoYO7lteQV1LV6JDEhEREUlK4YjH9fe/zZad7ZTtbOee18oTHZKIiIhI0qpt6eSWpevY1tDB+po2/v7GtkSHJCIiB4AKTMNIQ3t3z+dpgRQy0/b/j/+1LTv5z/paPM/b79cSERER2VdaO0M0d4QO6DUjnscza6t5q7zhgF5XREREZG/ZymbaQ5Ger1MD+/+aHd1hHnunkk01rfv/YiIi0ie1yBtGTjElPLhiO6kpAa46biI5Gfv3x//A8m3c9UoZ4QhcVD2Wj5wweb9er7f6ti7++lo56cFUPnTcRNKDqqeKiIjI0ORlBlk0rZiXNtaRnZ7Kl8+cud+v+YtnN7J0VTUZwRQ+cdIUzphzYFvybapp4ZG3KynNz+Ti+eMIBA7AyJCIiIgcEo6ZXMTccfnYymaKc9L5whkz9vs1v/PIWlaUN1KYlcYN5xhmj8nf79eM9drmnby6eSezx+Rz+uxRB/TaIiLJQgWmYeSKhRO56KixpKemEEzd/8WWjTWtRCevbKk78LNJfvXsJl7atBMAz/P4r5OmHPAYRPYnz/P43Ytb2FDdwjGTirh4/vhEhyQicsgIBAJ88YwZfOI9U8lKTyU1Zf8XW7bUtgHQGYqwrrr5gBeYbn9mI+urWwkAuRlBzj589AG9vsj+1hWK8LOnN1Db0sXZc0fznpkjEx2SiMghIyOYyncunENbZ5icjNT9PlElHPHYWudyp4b2bt7Z1nRAC0wtHSF+/sxGdrZ188L6WiYUZWFG5x2w64scCFVNHfzm+c10hSJcsXACsw5wEVcODlrSMcxkpwcPSHEJYNG0Yopz0ynMSuP4qcV7PO95Hi9vrOOd7Y375fqxS7PbusL75RoiifRGWT0PrdjBqu3N/O2NbdpXTURkHwsEAuRmBg9IcQnguKkjyMtIpTQ/gxOn7znw3R2O8PSaarbU7p+JOx3dLnfygKaO7oEPFjkIPfjmNp5dV8s725v46+vaV01EZF9L8XOnA7EKOjUlwILJRWSlpTKlOLvPSQNN7d08saaK2uaOfX79zlCEtm431tTRHaGxXbmTHHrufb2C17bUs6KikXter0h0OJKktIJJ9psTphdz1MRCIp5Hbh/t+O54YQsPr9xBejCFj500hTP38Szdy44ZTzgSIT01hSULtLJDDj15mUEygil0hCJkpaWQHlQrIxGRg9nF88dz5pxS0lJTyExL3eP57z9qeW1LPYVZaVx/1kzmjivYp9e/9JhxPPpOFcW56Vx41Nh9em6RZJCbGSSAK6JmHID9aEVEZP/69KnTuNLfAiKt12Tqju4wN/5zNZtqWxlXmMn3LppLUU76Prt2cW46lxw9jje2NjC9JIcFk4v22blFkkVmTL6Uoa1HpB8qMMl+lZ2+5+BI1MaaFjzcrI+1lc0DFpg6usN0dIcpzB56MnDY2Hy+c+HceMIVOajMGp3Px06awpodzSycWkReZlqiQxIRkb3U3//lnuex2V+51NDezdvbmgYsMLV0hCBAn5N8+nOKGcUpRvsHyKHr7Lmjae4IU9XUztlz1QJSRORgFwgE+h0nqm7uZJOfO21r6GBtVXOf3XWi6tu6yEpL7XOST38uXTCBSxdMiC9okYPIVYsmkRII0BmKcPmx+rsufVOBSQ4oz/N41tZQ39bN/ElFbK5rIystleOnjtjtuOVl9SxdXUVpfiYLJhfx06c20NQR4oJ5Y/jgwokJil4k+SyeU8riA7xHh4iIHDjhiMe/Vu4gMy2FoycW8oytoTQ/kxOn7z5AsnRVFcvK6pk9Jo+cjCB3vbwVAvDRRZM4dZaKRiLgBiLV2UBE5NDW2tnNv9+uYmxBJkdPLGTltkZmjsrlqAmFPcd4nsc9r1ewpbaVk2aMZHNtK/9aWUlhdhrXnT4DM0Z7KYmA21tNe9rLYFRgkgPqnyt2cOdLWwh7cPLMkfzqyqNIT00hJ2Z2red5/O7FLWzd2Q7A+qpmKps6AXh9S70KTCIiIjJs/Pq5TTy6qooAsOSYcfz6Q0eTmxHcbXZtbUsnf3hxC61dYZZtrWd6SS4N/j4Ar2zaqQKTiIiIDBvff2wdK8obyQgG+Myp0/j0qdMYkZ22237k/9lQx72vlxPxYEN1K9npKbR3h2lvDPPChloVmERE4qDmibLPlO9sG3TT6YqGNsKe+7y6uZOi7PTdiktRnrfr85K8DLL8QZTJI7P3WbwiIiIiieJ5HuuqmqlsHHjT6com97wH7GjqZGRuxh6tWzzPw+v5HMYWZJIagGAKTBuVux+iFxERETmwwhGPVdubaGzrGvC42mY3Qbkz5LF1Zzuj8jJ2Ky4BRDyvZ9wpgsfEYjfWlJOeyuyx+fs+eBGRQ5hWMMk+8eg7lfzhxTLCXoTLF0zg4vl9t54487BS1le10NYV5vTZfc+mDQQCXL1oEktXVVGan8E1J07hrDnNVDd3ctKMkfvzNkREREQOiN+/WMbDb20nOyPIZ06dxqJpfe8JcMZhpVQ1dZKWGuCMw/rOnUryMvnwcRNZvrWB2aPz+MD8cRw3bQQpgQALJo/o8zUiIiIiB5NbHrO8tGknpfkZ3HjuLCYV5/R53Htnj+LRdyoZkZ3e717f75kxkrK6Nsrq2jhxRjHvmVHCMZNqGVeYgRmtApOISDwCXuxSkV5qapr7f1Ikxi1LLS+srwPg6IkFfOv8Of0eG/07FwgEDkhsIiLDQUlJnv5TTQLKnWSovvT3ldiqFgDOmlPKp0+d1u+xyp1ERPYt5U3JQ7mTDEU44vGRO9+gvs21AP7wcRO55Jj+99TzPE95k4jIPjRQ7qQWebJPzBtXSFZaKunBAIePKxjw2EAgsNsb/fPrarjpoVX88rmNhCPKLUVEROTQN2dsPqkByMsMctTEwgGP7Z073b+sgpv+uYp7Xy/f32GKiIiIJFxqSoDZo92+SKPzMzh2StGAx8fmTRHP47fPb+amh1bx1Jrq/RqniMhwpBZ5sk+cObeU/Kwgj7xdyfaGDjq6w3vsD9DbivIGttW38eCKHVQ2dQKNTCjK5rwjxhyYoEVEREQS5CMnTKYwO8jysgZsZTPHTXUt7frjeR4vbKijrqmTe14vpzPk8fb2Jo6eWMiMUm1ELSIiIoe2r5xl+MOLW9i6s40V5Y39tsiLCoUjPLmmmvKdbTy0shKA8p3tvGfmSNJSNd9eRGRfUYFJ9pl/v13JWxWNAORkpHLNiVP6Pfb5dTXc/sxG2rsjZKW5N/YAkBHUm7yIiIgc+iKex2PvVLO9sYMVFU2MKcjkrLmj+z3+ntcruPf1ciIeBFNdISo1JUC6cicREREZBqqaOli6qoqOUITVO5o5YnwBU0b2X2T66dMbeMbWEkxx400eEExJGXBCj4iIxE8FpmGorK6Vp9fWMLM0lxOmj0xIDOurW2jvjgCQl5nGgsl5jCvM5PTZo1i7o4nl5Y0cN3UEUwdIFkREREQOhBVbG3izooFjJ49gztjEbPxcVtdKtJPwhMIsJozIZu64fCYV5/DKpp1sqm3h7DmjKcpJT0h8IiIiIlHPrK1hS10rZ80tZUxBVkJiKK9vByAUgZmlOYzOz+LkmSNJCcDjq6toau/m/HljNVlHRGQvqcA0zIQjHrcuXU/Zzjay0lLIyQhy5IQ9+/7HuyHiW+UNnGpGkpORSk56kCsWThzw+FNMCcvLGmho7+b02aO4/NgJAFQ1dvD9xyx1rd08s7aaHy+ZR26G/pqKiIhIYlQ2tvPDJ9bR2B7iuXW1/HTJPPKz0vY4Lp7cKRzxeG3zTs47YjQrKxoZW5jF4jmlA77mpOkj2VDdQiQCFx09llPNKABe3byTHz2xjvbuCG+VN/KDDxwe/02KiIiI7CMvbajl9mc30OW38/3RJUf0eVw8uVNbZ4iNNa1cPH8cG2pamTs2f8DVSwAnTC2mprmLvIxUPn7iVMwY11L4/mXb+OPLZXhAWV0b1y2eGdf9iYjI7jRyP4x4nkd3OEJDexcA7d0RKurb9ygwPfTWdh5asYP8rCBfPGMG44uyBzzvn14u44Hl2wimpvCRRZM4dwh7KE0ryeUnl82jKxwhO33XX8Ot9W3UtXYDUNXUSV1LV0+Bqamjm4ff2sHoggxOmzXwIIyIiIjI3vI8j8qmThrbQwDUt3ZR39a9R4HpN89v5uVNdYwvyuJr58wadB/K255Yxwvr6yjICvLlxTOZ18dkn95OnDGSBVOK8Dx2O/+W2taeVeE1LZ27DdZs3dnGc7aGw8bkMX/yiLjuXURERCRenudR0dBBV8gtu25o6yLiebu1pQuFI9yydB3rq1s4fFw+154+Y8C2daFwhJseXsPaymbGFmTy7QsPoyQvc9BYLj5mPOccMYa01MBuey5tb2jHXxROTUvnbq9ZWdHAWxWNLJpWzLSS3DjuXERk+FKBaZj4+xsVLF1VRXFuOqfMLGF5eQPjCrNYfFgpLR0hdrZ2MWFEFoFAgCfX1FDV3ElVcyePvlPFx07qfy8lcO3uwh6EQxHWVjYPqcAEEExNIdhrY8WjJxZxwrQRfqJRwMQRu5ZS3/b4OpZvbSQ1AKGQx5kD7FMgIiIi8m55nsftz2xkRXkDU4pzeK8pYX1NC4ePzWfiiCzqWjrpCkcYU5BFa2eIZ201zZ1halu6eGJ1Fe+bN3bA82+qaQWgsT3EWxWNQyowAWQE9yxcnT13NCsqGqlp6uS02aN6ikvd4Qi3PGYp29lObkaQb5w3i9ljEtPeT0RERA5tXaEI3390LWU72zhyfAHHTCqkuqmT984uISUQYEdjO5nBVIpy0lm+tYGXN+0E4Flby8VHj2dicf8Tm+vbutlY0wLg9q4sb+SMwwYvMAFkp++ZO501t5RNta20dYdZfNiuycvb6tu4del6Gtq7eX5dLf+3ZB456qgjIjIo/U85DFQ1dfDA8m20dIWpau7kiPEF/OKDRwGwvGwn337E0h32mFmaw22XzGNUXjqba1tJSwkwacTgvXKPnljIusoWgqkBjplUtFexpqYE+OrZs/p8bqe/sinswbbG9r26joiIiEh/Xt1UxxNrqvE8qG7u4suLZ/CFM2YAcM9rW/nr6xVEPDjFjOTa02ZQkpdJc2cr+RmpTB81+GzXeeMLqGrqYGRuBsdP3buVRflZaXzvorl7PN7ZHaG2xa1ab+kMsaWuTQUmERER2S9+95/NvF7WAMALG3by6w8dRVG22xfye4+u5aWNO0kJwFXHT+KEacUU56RT19rF6IJMinMH3j9yRE4688YXsKysgWklOSycsne504zSPH68ZN4ej5fXt9PQ7sadapo7aWzvVoFJRGQI9D/lIc7zPG5Zuo6WrnDPY03+G2Z7V5hbl66nO+wWB2/0Z9N+afFMHnprB6MLMnnPjJGDXuOio8bxnhkjCaamUNCrZcwTq6tYvb2ZY6cWcfzUYrpCEf70chnt3WGWLBjPqCEsa35qTRX/frsKD48xBZkUZadx7uFDWyUlIiIiEo/G9m5+/cIWPG/XY6Gwa0FXvrON+97YRsR/7o3N9aSmBLjhHMOztoZZo/OGVMT55MlTufDIsRRkp+3WKtjzPP72xjaqmto5a+5oZpbmUdvcyV9fLyc9mMJViyb1uYqpt7tf3cobW+opzk0nPxxhQlE2p80aFd83QkRERGQI3ipv4Om1NT1fhyNh2rpCFGWns7ysnlc2utVKEQ8eX1XF+48ex/+cY3hzawPHTx0xaBEnNSXA18+dTWVjOyV5maQHd3XC6QpFuOuVMjq6Iyw5Zjwj8zJYV9XMI29XUpqfyWULxg+6z1M44vHzZzayqaaFiUVZdEcizBtXwJiCoa2SEhEZ7lRgOkTd/UoZD63cgRfx8Ho9t766Bc/zelY1ReX6AxyZaalcesz4uK5XnJuxx2O2spnfPr+Z9lCEN7bWM2dMPn97o4J/vrUDgMa2br5+3uxBz33/8u2U17sVSxOLMvnKmTP7vJ6IiIjIu3XLY2t5fUsDqQFo9fc0inp1SwMnm1Hct2wbXZFdmdXIPDfjtjQ/kyULJgz5WoFAgDGFe64Sf+ydKu5+dSsesLm2jR8vmcdvXtjc00YG4OPvmTrguetbu/jnih20d7sc74jx+Xz1bLPb3gMiIiIieyMc8fjqAyvZVNNGWgp0hHblR11hWPpONR8+fiL3Ld9GbFY1yW+FZ0rzMKV5Q75eakqAcX3sD/6nl8t2jTG1d3PDObP45bOb2FDTSgAoyApyziATlF/cUMsTa6oBSAnA+44YwzUnTh60MCUiIo4KTIeg7z2yhpc21ff7fE56kM/f+xZldW09j+VmpPKTy/ZcIrw3WjpCdIRcKtHRFaYrHKGje1dBqzMU6e+lu8nN3PXXdGt9B3e9spVrT5+xT2MVERGR4cnzPP77rmVsa+ra7fEA9EzSCeDxsbuWUdey65jSvHR+dOm+zZ2aO0M914zmTLH5UscQcqeMtFTyMlN7CkwrK5q4f/k2LoujACYiIiLSn87uEFfe8Rod/vBOdN5yNHdKDUBDWxcf/sMbtHaGel43a1QOXz3b7NNY2mPGmHrnTh7Q0hnu62W7KcnLICsthfbuCBEPHnprB4eNyWfR9OJ9GquIyKFKBaZDSHtXmLtfKeu3uDRnbB6Hj80nFPG4b/n2nsfHFmTwpcUzuPeNCjq6w1y+YCJjCoe2FHh7Qxu/f7GMcMRj8WGl3L98G80dId5/9DgWHzaKC44cw/rqFo6eWMTI3AwuWzCBxo4Qnd1hLjt2aAMd1542nW8+tJodTZ0AaBKJiIiI7At1LZ3cutTuUVwCNyhx1IQC5o0vYGt9OzXNu44xpbl88j1T+NnTG8gIpvLREyaRlT60tHpleQP3Ld9Gdnoqi6YX8/c3tgFwzYmTufDIsZTXt1HT3MVZc92m05cfOx4PyEgNcNmCwVeYZ6en8vnTpnPr0nU0tLtBHaVOIiIisi9sqW3lG/9c1VNciuUBp5hiDh9byNNrq2nu2FVcWjRtBOfMHc1tj69jVH4mVx43kZQhDu48ubqKZ2wNYwozmTwih4dX7iA/K8gXz5jBkgXjaWjrpjMc6RljuvzYCTz2TiXFuelceOTYQc8/e0w+/3XSZH7z/BY6QxFSApCihd8iIkMW8LzeDdR2qalp7v9JSSoRz+O6v73Fhpq2Pp9PDcAFR47lIydMZtmWem7+9xr8rZc4ZeZIcjOD/GtlJQALpxTx9XMHb10H8MPH1/HculoARudnUOkXgWaU5PCjPjZNjGrp6GbVjmbmjs0jJyOt3+Oiqpo6+PMrW0lJCfCRRZMozB54E0gRkeGmpCRPY8hJQLnTwaOhvYvP3bOC+rZQn89nBFP4zKlTOcWM4h9vbuP3L5b1PLfkmPFsrm3ltS1uUs95R4zmE4O0rou6/v63Wb2jGYDR+elU+sWtRdNG8D9nz+r3dbUtnWyqaeWoiYVDane3ensTD63cQVF2GtecMJmgWuSJiPRQ3pQ8lDsdPFZta+Smh1fTGer7R5afGeSWi+cyrjCbnz69gSdWV/c89+XFM7lveQWba92Y1SfeM4Xzjhh8b+1wxK0ij070Kc5Jp67VfX7hkWO55sTJ/b52a10bO9u6mDe+YEjt7p6zNby8qY4pI3Pian0sIjIcDJQ7aQXTIWLN9sZ+i0s56Sm0dkV4ck01AcCMzuPa06fzwJvbGZufwVXHT+LeNyp6ju8ODz2/y4jZXDE3I5XUAIQ9GJHbfwGorSvM1x5cxabaNqaOzOb77z+crPSBN6wuzc/kusUzhxyXiIiIyEDufXVrv8WlzFTXju6e1yqobupkzrh8PnjseF7aWMfM0jwuPWY833lkbc/xoThyp/Tdcqc0wA2SlOT1v7/k5tpWvv2vtVS3dLJgchHfGMIeloeNzeewsflDjktERERkIL9+blOfxaVgAAhAU0eIHz+xgUXTi7ngiDF4EY/Nta3Mn1TESTOKufeN8p7XdA1xy4RAADKCbrwoNQD5WUHqWrtIDcC4ATrv/GdDLT9/ZiOtnWHOPXw0nzh58IlAJ5sSTjYlQ4pLRER2UYHpENAdjvCth9f0+dwR4/LZUN0CuDf7+9/cTlpKgGtOnMxPLzuy57gPLpxAa2eIzlCEDw6hdd36qmZ+9fwm2jrDHD2xkJLcdK5YOIFlWxuobe7iggGWIW+qaWGTP2tlU20bm2tbNQAiIiIiB8z2hjb+9U71Ho+npQaYNz6fN8oa3XGNHdz1ajl5mUGuP3Mmlx87sefYDx03kWBqgPTUFD64cPDc6cUNddzzWjkEPI6dUsTYgiyuWOhauAQCgQFn8S4va6C6xa0SX1/VTHc4MqRVTCIiIiL7wjNrq9m8s32Px7PSUpg7toDXy9yq7g01LdiqFh7Nr+QHHzicETm7Jh9fedxEHnunipG56Zw/b/DVS/cvq+CJNdVkBlM4cXox00pyOGvOKB5eWUlpfibvnTWq39eurGjs2X9pXVVzvLcrIiJx0G+mh4Dali7a+5hFkpmWwvVnG042JRRkBslKcz/u7ojHE6ur+d1/NtPU0Q1AUXY6XznLcON5s5k2KnfQa/777UrWVbVS0dBBKOLxmfdOpygng9Nnl7JgchEvb9pJKNz3jJTpo3KZVequMXt0HtNG5bzbWxf5f/buM0CucHCXZgAAIABJREFUszr4+P/eO31ne9eutmhXvUuWLFlyxZY7xsYYUwwBQ3ollIQekpCETmgBA3kJBAI2BhtjLLnLkm3ZarZ62d7bzOz0du99P8y20cxqd1VWkn1+X6K997lNON7j5zzPOUIIIcSM7Wv3ZT1ekWfnEzct4rKaAvIdFkZCJwLRJD/f1c7Pd3WMxTeNZW4+c+tiPnHTQgqnUbr39wd6aPOEaRuKkGe3cP/mOhxWjbetrqKu2MWrLR4mK129oaGQOfmpHU7LqvIluSSEEEKIWXWgazjr8bU1BfzldQ3ML3OTY1MZnQbq9cf47rNNPPZ6z1h8s3FeMf/01iX81XWN0yrd+8ShPrp8UZoGw9QWubh7bTVuh417183Fpqkc7M7+TgDr6wrJd1qwaAor5xbM/IOFEEJMm+xgegMoy7XTWOrKKJEXTRj86f/sIZw00A0odFmxWRRiCZOmwRBNgyH6/DE+ecvk9f5P98xRRTnjPZT2tfv4ytZj+GM6u1qGsvYSaBkMEUnqlLhtLCh3s/VQHzcuLR/b9iyEEEIIcT5d0VjCr17txBNJL5HX4Y3yoZ/sJhDTMcxUnf+ErhNJGBzuDXK4N0gsqfOBTXUzfubEFbylueMlXR7d381PXmojaZi8baRf5qlO9AUxTKjMt1Nd4OCpw31ct7hs2s2xhRBCCCHOxg1Lynnu2ACJU9YR72zysL/TRzhmYAAlbivxpEkolmRXq5dXWr04LCrXLymf8TMLc2z0+mPYNIXqIufY8e9vb+HxA73YrSof3lzPlqWZ9z7ZH8KiqtQU2nBZNXY1e7h8XtGM30EIIcTUZPnjG4CmKnzlHSu5Z23mFuNA3BhbQeINJ7hj5RxiE2rdHu+dfKtw62CIB15o4cnDfRnn7l0/lz++so73XD6Xv7y2Yez4we5h/CPbkFsHs/eE2na4n7ahCIPBOL97rYcHXmjlS1uPT+tbhRBCCCHOVqHLxg/ev5b5pa6Mc8PRVHIJYCgU5/0ba9P6U56uzMreNh8PvNDCnlZPxrm/uq6B91yeip/uXV89fr/+IHHdxDChaTCU9b5PHu6n1x+jZzjG/+3u4pvPNPHjHa3T/FohhBBCiLOzuDKPn96/nlx7+sJgEwiOJJdSBxRuXFLKaOhkAp2+zNJ6o54+0s8PX2iheSAzBvr4lvm8Y20Vf/2WRjY3lowdbx4MYZJaVH1skrjs6aP9DIXiNA+G+cnL7fzH1mP8/kDP9D9YCCHEtMkOpjcITVW4b2M9cR22HerDJNX8sM8fHxuT77SwvCofk/HGioU52Uu6mKbJ1548QctQGJum4LZb2NhQPHZeVRRuX5nZZ2lTQwk7Tw4xGIyztrYw673n5DtQSAUao0FIty86008WQgghhDhjdovGV+9ZyRcfP8r+Dh9WTUWBsYUyAHXFTk7dJDRa5vdUQ8EYX3/qBL5IgueODfC1e1ZQnje+U8lu0bh3XWavpo3zijnQOUxCN7i8LnvsVJ7ngC5/2rEOb/aFPEIIIYQQ50OO3cKP/+gyPvXbg7QOhsl1WAhEk8QnLMSpL3ExGEqkXXf9otKs99vV7OG7zzcRT5rs7xjmP9+1Mm13dkmug/dtrM24bn1dIe1DYZw2jY2T7Eoqy3PQ64+NzT0ldJOWSRbyCCGEODuSYHqDuX9zPfdvrsc0TT7z24NpCaZAJMkrrV42NxbzaquXXIeFL9yxJOt9DBMCsVTZmLhu0uufXgJoXmkO33jnSkIxnWJ39uTV3WurKMyx0eEJs7fdizecZFNjcdaxQgghhBDni6IofOrWxUBqcc2f/GxPWoKp3x9FN2F+mZs2T5iqfAf3ZZnoAPBFEmO9LQPRJJ5QPC3BNJlNjcWsnJuPbpjkO61Zx/z5NfOoKXLSH4jxSosH3YSr5mefrBFCCCGEOF8cVo2vvmMlANFEkvt++Era+ebBEH929TwOd/vxRZKsrM6juih73+3+QIz4SD/xQDSBbpio2tTlf+9eW831i8uwWVRctuzTmp+4aQFPHOilNxBld6uPHJvGdQvLZvKpQgghpkmZrJkwwMBAYPKT4qKiGyZHewNU5jsIx5N89pHDDATjGeMaS3P4+jtXTuuev9nXxXPHBinPs/PRLQuwWc59RUXdMEnoBg6r9F8SQoizUVqaK81YLgISO106YkmdY71BGstyONwT4MtbjxOO6xnjNjUU8Q9ZekqeyjRNfrC9hUM9fhZX5PKnV89DOQ89khK6gWlyXuIyIYR4s5C46eIhsdOlIxBJ0OoJs6gil62H+vjRjlaSRub/fG9bVcn9m+unvF9CN/jqthN0D0e5an4xd6+tnvKaMxFL6lhUFU2V/7cXQogzdbrYSXYwvUF87ckTbD8xSHGOlaVz8jKSSzk2lXDcYP4kZV2yuXN1FXeurjqr94oldX6+q4OkYfLu9dXk2NNX5mqqgqZKckkIIYQQsyehG3z2kcMc7glQW+ikLN+ellzSVFBMUDWFRRW507qnoij8ydXzzvrdBgNRHtrbTZ7DyjvXVWdMhlg1SSwJIYQQYnb1+aN87tHDdPmiLK/OIxxLpiWX7JpCTDfJdVhYUZ0/rXtaNZV/uHnhWb/bsb4ATx/pp7bIxa0rMnuT2y0y5ySEEOeTJJjeAEzT5FC3D4ChUIKdTUNjdWYBKvPtfPLmRQwGY5P2RTpffryjjccP9gKpEn0f2TJ/Vp8vhBBCCHGq3uEoh3tSTaHbvBH6ArG088vm5HHvurloqsriyuklmM6V7zzXzO4239jP7748s2+TEEIIIcRseqXFQ9dI7+wDnX5y7elJmxuXlrNybgEV+Q5qilyz9l6GafKtZ07SNhTBokK+08rm+SWz9nwhhBCSYHpD0A2TYFSf8DOsqs7DbbdQlmfnPZfXYLNo1JVkr3sLqSTVvg4f+Q4rDWXT3+U0lVA8Of7nRPI0I4UQQgghZofTqqICxsjP0aTBhvrUIpzFlbncubpqyvJ2Sd3g1VYvdSUuKvOd5+zdJu6kCkQTpxkphBBCCDE78p3pCaVATOempWX4Igk2zSvmmkVT9zcKxRLs7xhmWVUe+c7sPbtnyjQhNNI/M2mAJ5zZKkIIIcT5JQmmS5wvHOPD/7OXmJ5e9/ZQt5+bl1WMJZem8qMdrTz6Wg8Oq8afXl3PdVMEB9sO9fLgni5Q4PpFZbxzXebq2p7hCLcuq2A4ksAwTN6x9uzK7QkhhBBCnK2WgQB/96sDY8mlUYe6h7l33VxuW1E5rd5J//7EMXa1eCnOsfKPtyxiYfnkO50M0+SnL7Xx/PFBHFaNey6r4ppTGk3rhknvcIQ7VqVKu+TYLbzjsvPTi0AIIYQQYrqePdLL155uzji+v93HBzfXsWFe8ZT3iCZ0Pv3IYU72h6gpcvJvdy4jz2k9zfgk33m2mQNdfgpdVv74qnoWV+aljUnoBn3+KHeunsP244NU5Du4eVnFzD9QCCHEWZEE0yUsEE1w34/3ZD2XMODR13t54lAfVzQU83c3zEc9zWTJ8f4gJhBJ6Bzq9k+aYDJMk489dIDjfcGxY08c6uPutek9An60o5XHDvTgtmkUuKzYrRqxxKlTOUIIIYQQs+doj5+P/fpg1nOBmMEDO9r42a4Obl9ZyX0baie9j2maNPWHgFR54v3tvkkTTKFYkr/6v/0MBMZX1D5+sDctwWSaJv/xxDF2tXgozrHhtluwaArxpMROQgghhLhwHtnXwQ93dmQ91xuI88U/HKfAaeH9G2u5fkn5pPfpGY5yciR2avdEONobYH19UdaxrYMh/uHhg4RGdnUPheI8dqA3LcEUjiX57KOHOd4XpDLfjt2qEU8aGKaZ9Z5CCCHOH+kSfAn70tbjU46J6ybPHR+kbTCU9XyHJ8y//P4ogUgCqwIFLivr6ibv09Q8EEpLLgEUuWwZDaj3d/hI6ia+SJLWoQjHeoN87/lmTPllL4QQQogLwDRN/v2JY1OOiyQMnjrSP+kExf4OH5//3RE0FRRgToGDjQ2Tr9x96kh/WnIJoCjHnvZzKK5zoHMYw4SBYJyWoTCvd/r5z2dOTv1hQgghhBDnQTxp8JOXO6cc54sk2Xa4f9Lz2w738aMdrRTnpMriLSx3s7wqf9LxTx7uG0sujSp1p8dOr3cNc6wvtVC6ezhGy2CYl5o9PLC9Zcr3FUIIcW5JgukSNRyOc6IvkPXcwnIXE9M9LqvKQ3u6+fhDB/jDgd60sd99rpldLR46fVGWVufx7XetytjeHInrvNrqIRhNUJHnoCRnfBvz4opcPn7Tgox3aMzSx6nLF+Xzvzs8g68UQgghhDg39rb7GApl9jRSgAVl6c2oNeCLjx/lHx8+yN42b9q57zxzkr3tPvoCca5fXMrX71mZ0czaE4qzp81DQjeYX5aD05KKzFQFNjcU8bdvaUwb77JpzCvN7JV5oMvPz15qO4OvFUIIIYQ4Ow/t6SChZy64sShQ7k4vb6cqJp/+7SE+98gh2j3hseOxpM4Pt7fwWucwQ6E4922Yy7/dtQynLb2VQ5cvwuudPkzTZG6Ri9GzVlXhbasquW9DTdr4xZW51BRm9sB86mg/LzcPneEXCyGEOBNSIu8S9Ou9Hfzvyx1MVnGu3RtjNASwaQpFOTa2nxwEoNMb4frFpQTjOjtPDtE6NL6zaSgQJ/+UGrixpM6nHznE8b4g9SUuvvi2pXz+rUvY1eJhbW0hDaWZiSSAv7qugbhusP34YNrx/e3DdHjCzD1lIkYIIYQQ4nz52pPHee7YYNZzNotKy1Bk7GeXVcFqUdnVkkosDUcSfLe2kA5PmL3tPgZD47uRhiNJXKdMkHR6w3zhd0fo8cdYW1PA525fzKdvW8LRXj9vWVROsTuzqbWqKHz29sX80++OcKDLn3buySP9vGdDzbT6QgkhhBBCnAsff+h1jvQGs55z2jT6guOLdgpcFgaCcfpHdmybL7TwhTuWcrjbz66WISITSv5aVRWrlr7W/ZUWD998+iSBaJKblpXz59c04LJbGAjEuG1FBfYsfcXznTb+5c6lfOa3h2jzjMdxugG/3d89rb5QQgghzg1JMF2CHtrTPWlyqTTXNlaGpTTHStww6PRFx84HYkl+93oPL5wY4uRACOuE0nZrajNL43V6ImMl8VoGw3x523FK3Hbu21BDgctGNKHzi1c7wIR3rZ+Lw5r6xa8qCneuqmRX8xCx5PiKlwKXNSOJJYQQQghxvgxHEjx/fJDJivSW5NjoGk7FSnXFTvr8MbqHY2Pnu3wRnj7cy89f6aI/GMOiKjByt82Nmb0Ddrd66fGnrj/S6+cr205QlGPl/RtrsWgqA4EoD+/rptBp4+7LqsZ6ZNotGpsbiznc40efEOcVu22SXBJCCCHErHmlxTNpcklVUzuvA7FUCbuGEgdtnhhJIzk25nCPn2eP9vODF1oIxnRGIyebpmQtK7y/Yxh/NHX9vjYfX912nNpiF3evrQbgWE+AZ48PUF+Sw41Lx/s8FbpsLKrMpd0TSYvz8h0y5ySEELNJEkyXGNM0Ccb0rOcUwKqAw6IQS5oMZCkDA9A6FKFtZMtywjBZXOlmYXku79uY2cy6ptjFkjm5HO4OkGvX2Ns+nLpON/nIDfN54IWWsVq7nlCcO1bNQVHgRzta6fZF0pJLmxqKuHl5BXmSYBJCCCHELBkOxzEmyS5ZFMA0sSqAkoqRTmWY8Gqbj/7gaNLJ5Ip5RTSWu7lmYVnG+I0NxWw93E+nN4KmKGw/kdo55bJZeNf6uXzrmSb2daTiqUgyybULy+j1RfjVni56h6NjySVNgRuXlnP7ysqz/BsQQgghhJi+o73Dk55zaspYMsdpgabBaMaYWNJkX7tvbO4q32lh1dwCVtcUUJHvyBi/rq6QnScH8YUT+GNJnjs+iKZAeZ6DKxqK+eYzJ+nwRrBpCoZpsqamgF0tHp45OkC3bzy55LCo3LK8gnesrTrbvwIhhBAzIAmmS8zbvvNS1uOFLivecIJuf2r3kqbAqaVyVSDHrrGwPId40uDVVg+aolDssvLeDTVYNZXXOn08c3SAmiIXb19ThVVT+Zc7ltI+FObBPZ3sbPIA0DQQwjRNQhOSXS81e3ju+CClbhsDwfRm1gCrawpYWV1wbv4ihBBCCCGmEE8a/MUvXst6rijHiieUoGtkt1G2LU4qkOeycOOSUrzhBMf7gzgsKqW5Nu5eU4WiKDx7tJ/9ncOsmpvPtQvLKM9z8NV3LKfHF+W7zzcTGNkJfqw3VfpuYtPqR/f38Nt9PeQ7rQyF0mMnw4S71lRRnpc5ESOEEEIIcT50ekM8uKcn47iqQL4zNe8USsRRgEgy83pNgdJcBzcsKaNpMES3L7XgprHMzXWLyjBNk4f3ddPuCXPD4jKWVeWzuqaAb967kqFgnH/+/VHCcR3dhOO9AdbXFxKMpR4U102+v70Zu6aiqSqBWPoLaKrCBzbVnYe/FSGEEKejTj1EXCx+vKOVbJXxVAVuXV6RdixLH0YMIBDTefS1XnLtGnHdJJI02NHk5Tf7utENk+8918wzRwf46UttbD8+AIBVU2koc7OsMnfsXu2eMH842MdbV1ZQ4rbhsqnERurqesLjEySFLgs5No1Vc/O4blHmKl8hhBBCiPPlHx4+kPV4jk1jXZbSwKcyAF84yYN7ukExSeqpneSPvNbLy80eurxh/mt7C88cHeAHz7fQO1Jqz2Wz0FDmpsg1vmt7T/swR3v9bFlcRoHTgtOaisWShkkgOqGPgdOC266xZUkZZbn2s/sLEEIIIYSYgY8+dDDr8eoCJ/UlOWM/T1Z6WDeh1x9l2+F+grEkSQOGwgl+9nLbWNni/3mpjWeODvDd55oxzNSdClw2Gsrcqd3lI7Ye7sUwYHNjMfkOCy6rim5AOGEQS6YW7KhAvsNCnt0iO5eEEOICkR1MlxBvOHNXEEBxjo0rGop5rcPHge5A1jGqwlh5mEhc58mj/ennMTFNk/hIkkg3ySjFt2Nk99LE92kbCjM4sltJA3RSTRXL8+xsaijm3nVzcdoyGzIKIYQQQpxv0UmaVs4pcHDn6ir2d/joC2SPr2yaQnxkxc5gKE7PhL5MAA6rSiRhkBiJnaK6QSSRHjs1DYTSfvaGErzc4sE3suRXJZXEiusmdcVOrmws4e1rq9FU6bkkhBBCiNlnmtlTR1WFTu5eW8XRHj/hSeIr50hsBNA0EMQzoW1DQjexqBCMJcfmpmJJHdMk1e9hhDcyfk00YRJL6rzeNcxwNDlxGAk91e7huoVl3LQsfcG1EEKI2SU7mC4hf3VdI7n2zGTNQDDOd55rypjEgNT2ZEgll5bNyaUy385ldYXYtfT/6Z8+OohFU7l33VyWV+Vx45IyPOE4//HEMXY1DwGwvn58pa9VVbhr9Zy0FbcTp1TiSYMPbKqT5JIQQgghLphP3boQa5Zo90R/iJ+82Jq1pO9obscwTJbOyaUi386KqjxOzfn8dl8PjWVu3r62iuVVedy1eg5bD/Xx5a3HaB5IlcVbWOEeG59r19gwr4hAdLycy8TpGYdV4551cyW5JIQQQogL5i+vbcg6UfhSs4ff7OueNLkEqfmnRRVuqgucLCzPTTunm/DL3Z3cvKyCLUvKWF6Vx+0rKvnPZ07yjadOji2ori50jl1TU+Qk12HFP5J0mpj6MoHKfKckl4QQ4iIgO5guITaLyudvX8zHHjqYUSqvyxs57S96gObBEOG4wUCgn+pCJx3eyFgj6cFQHMM02bK0nC1Ly3n22ADfeOoEhgkn+4OsrS3krjXVuO0WTvaHeNvqOVg0NS3BNFFZnpR0EUIIIcSFVVXg4k+vbuBbzzZlnDvaFxxbQTvR6LGkCW1DYYIxHV94iMbSHI73jy/mafOEAXjP5TUA/PSlNn5/oBcAbzjBF+9cxsduXEhdcSf+aJJ3r68mEE0SjmdpWABUZml6LYQQQggxm66cX8rrnT6eODSQcW5/x/Bprw3GDbq8EQIxHV84RkOpi6aB8Nj5w91BNFXhr65rBODLW4+x/URqQbNhmnzkhvl86e7l/OzldhxWjXvXzaV3OIppZl98s6ZGenwLIcTFQBJMl5BoQuczvz2UtQ9TbJLkkk0bb7wYjqfGJA1oHYqkjcu1a3znuSZuX15JXUkOoVhibILFF0mMrRTZsrSCLUtTf/7Xx4+yv9Of8UxNgQ9fWT/TzxNCCCGEOKfahkJ8J0tyCSCayJ7osakwEjKNlQuOJoy05BJAvkPju8818Z7L55LvtDEUGt8N1R9IldNTFYV3rpsLQEI3+PivD9DhjWY802VV+aA0pRZCCCHEBbbj5GDW5BJAUtezHlcY310UGImdgnGD4ITkEoDDqvD/drby3g01WDSVgcB4+eGBQCo+sls07t+cmk/qGY7w2UcP44tkLmwuz7WxubFkJp8mhBDiPJEE0yXi9/s7+K8dHZOejySzJ5gcNiuRZPZdRhMNhRJsO9TP80cH+Oo9K9D18SW9pglJ3eSFE/10eiPcvLSMwVBirPwLQJHLyoZ5ReiGya3LK6kvHW/+aJgm33uuiQ5PhCsainnrqjnT+WQhhBBCiDP2jw+9xsHezPLBoyKJ7D0GnDYL8Wj25NNEzUMRmoci7Dw5xAP3rcGuja+uTfW+NPnda92E4jpvXVnJnjYvrYPj71NV4GD13HwME969voZ8l3XsXCiW4D+facIfSXLHqko2zCuezicLIYQQQpyxd/3gZYLxySvjRJPZYye3XRtLLFmU1C7wbF7r9PNap59XWr18+92r0lo3WFSFaELn4b1d5Ng13rK4jMcP9I4t2lGAJZW51Je4sFlV3rO+Nq2scLsnzI92tGICf7SxhnmlboQQQswOSTBdAoLR+GmTS6fjDacnl0abSU8mppv84WAfVy8oIc9hwR9NUlPkZFfLEN9+tomEbvLkkX58E+7rsqp8+tZFzD+lxu6oZ48O8MShfgA6fVGuX1KO6zS9mXaeHOLhfV3k2DT++roGSnKlZIwQQgghpu+1Du9pk0unM3xKcklTUn0DJuOPJnnm2ACN5blYjw6Q0E3qinP4+a4Ofrm7E4AnDvalxWRFOVa+fPdych3WrPd8cE83LzZ5AIgk9CkTTA/v7WL7iUEq8x185Ib5WDVpsyqEEEKI6fv208dPm1yajML4riVIJZemip06vBH6/FHmlbnZ35WqijOv1M23nmli+4lBAB7a0522c2leiYsv3rUMVcleLu+Xr3awt90HgMOi8slbFk36fNM0+f72Fo70BFg2J48PXVmHMsl9hRBCTE3+6/MS8J3nWs7oOmeWrtbTCRf2tHlpLHPzjzcv5L4NNXzqlsV0+aIkRiIE3ylJq3DCGJtAySbfacE2sqrXaVOxTNG8+uF9XRzvC7KvY5iH9/VM442FEEIIIcZ9dduJM7ou1565AOZ0EySjXjo5yA1Lyvn7G+bzgU21/P2W+QwEx8u+nLrgxxNKnDZ2ynWMv4fTOvmiHEiV3ntkfzdNAyF2nBziiYN9U7+wEEIIIcQEW48MntF12eadphM7Pbq/h/dfUcufXzOPv7hmHu+/ohZveLzc8Kll8VqGwmw9NHmM45gQLzmmiJ2O9gZ4/EAvzYMhfn+gh+bBM1uUJIQQIkUSTJeAHSeHzui6KxuKsGqnT+bMybdTX+wi3zm+mc0bThCMJllWlc89l1VT7LZx+4pKFpW7mexuvcOZ/QRGXVZXxAc317Gowk2u3cIvXu3ANCePOHId4+9SlJN9Za8QQgghRDbReBJvZOoSd9lcu6B4yuB4XomLqgIHufbxkaPP29RYwl2rq7BbNG5fUUlFnn3S+3Rn6cU06m2rqnjXurnUl7gwTJPHXp98wY2mKmOxk82iUJY7+TOFEEIIIU51tNt7RtepwJXzpy7ju7DcTUWeHbdtPHbqHo6gKgo3L6vgpmUVqIrCrcsryXdkL7RkmNA6GM56DuD+TXXcuqycuYVOvKE4r7R6Jh1blGMjb2QOLNdhJd8p805CCHE2pETeRe5bTx0/42v3d/nHdh1lU5xj5WNbFtJY7iYS1/nEr1+nwxulutBBgSv9F2ye08qHr6rnYw8eYOIdLapCnsPC1QtKT/suVy8o5RevdDAcSdI0EKKx1M2mxuyByN9c18iv93ZR4LJy15qqaX+vEEIIIcT9P9l9RtepwI4m72l3e9cUOfjincvIsVto94T53KOHGY4kqC7MLOfbWObmrSvn8IMX0nei2zSFwhwb1y6aPHbSVIWFFW5+tbsD3YT2oTCbGospdNky31tR+Lsb5vP0kX7qS3K4fF7RdD9ZCCGEEIKPPXzkjK5zWFV2nkxP5Exsy6AAK6vz+ae3LkZVVV5p8fD1p06Q0E0aSjJ7JG1qLOZg1zCPHehNO26zqFTm27l+8eSxk8tuoSTXToc3Qoc3gi+SYH1d9pioPM/BX1/XyP6OYdbWFlDilsU5QghxNiTBdJE70BM442v7A/GMYypg1RRiuslQKMHWw300lruxW1WShknSMGkdDLPz5BCb55fgC8f5xlMn8IYT3LCknPI8Oz3+VMkXp1Xls7ctZmFF7pS1/lWFsdJ4mqJgs0y+s6owx8aHrqw/4+8WQgghxJtXNDHz/gGjPKeUsoNUHwFFUUgaJu2eKFsP9XHXmipsmkokrpPQTQ50+ekdjlKR7+B4n58f72gjrpu847Iq3HaN4EhvgkKXlf+4axlleY60xtTZ2DQVTVXQdRNNU9BO0xugodRNgzSzFkIIIcQsiusGyVPCLoPUnFNSNzGB17uGOdjtZ0V1AbGkTiSuo5vwcouH926sQVUUXjg+yMP7u3BYVO5dW83Ww31ji6WrCh187e4VOGzapP2XRk2cl5pq7Pr6ItbXy6IcIYQ4FyTBdJH7xA2N/O1Dh87Z/QxA04CRHowvNg3ycvMQVYUOunypUi26CcORBLtbPXxl2wlC8dTgB3d3pDVvXFyZx7IAxL7jAAAgAElEQVSq/KzP2d3qYU+bj4ayHDY3luCyWbh/cz0vNQ3RWOZm3SQrSYQQQgghzsZdqyv5vz0z7+E4WVpKN8FtU8eSRL/e28Uj+7spzrGOxUjRhE44nuR3r/Xw3y+2jk2K/OTFtrHrAK5oKKaywJn1OdsO9dE8GGJtTQGrawpYXp3P+zbUcqwvwLr6QvKkfIsQQgghzoP5JS5OnKb83GROTS6NSugmDotKNGmgqfAfTxzHYdVQMcf6M0UTOqYJD+xo4fev94wd/0G0Na0Szy3LKnHZM6cudcPk4b1deMJxrltYRkNZDretqMQTjjMYiHHL8soZf48QQogzIwmmi9xMkks2TSWpG2kTJHaLQiyZXiavyGXlinl57G7z4RvpGeCLBNPOn+wPsbNpaGziBMCiaiT01HhNhQ9uqs36HgOBGF9/6iT+aGrsr17t5DO3LebK+SVcOb9k2t8jhBBCCDET8WRyRsklp1UlcsqOJ4dFIXpK7FRT6KQ838FLTZ6x+GbibqeyXDu/P9BL00AobVJkYsvJPIeFD26qy/oeLzcP8f3tzcR1k98f6GVxhZsv3LGUO1bPmfa3CCGEEELM1NFu74ySS06LSmRCZskysmno1GTT0spccp1WdpwcxB9NjsVPo0rdNr6/vZlXWzxM7OygTwieqgud3L6iIut7/HpPFz/d1Q7AY6/3srmxmI/fuIAPXFE37W8RQghxbkzVx1hcQO/5wUvTGqeN7PzV1PHVt6M7g09NLllV0DQLO5s8Y8mlUwViSZ462k9Tf5DRSnbVBQ4+fesCrphXRFWBk9uWV1JbnJP1el84QTA2fu8ef4ydTUPT+hYhhBBCiDP19v96ZVrjRgNgdUKZutFSvqcml/LsCv5okhebPESzLNVVgJ7hGNsO9zMYjI0dW1KRy6duWcjyOXlUFzh4z+U12CzZQ+/+QIz4hNmVI71B9nX4pvUtQgghhBBnwjTNGfVfUgBlQihjUVOJpVPDo5pCB13DUV5q9mTd5WRR4XBvkD8c7EM3zLFjV80v5i+umcf8shzqi518aHMdyiSl7nyR9JYQ+9p9hCbsGhdCCDF7JMF0kYomkvjj5tQDGZ8QGV2BqwBkudSqwvs21tA2FM5YrTtKBZIjExzhhMHoHEuXL8oTh/q5Y9UcrJrCq21enj8+kPUejWU53La8Epct9Y9XvsPC8jl50/oWIYQQQogzcbxneNpjNS31f0cnIlQFkkZm8JTntHDzsko6fVFik9SBsVvUsbArMLI61wRODqSSRFuWlmGYsPVQL8d6s/fWvHlZBRvnFWEbWSFUVeBgUXnutL9HCCGEEGKm/ntny7TGKaTmikwgHE/FQxZVyZo8mlvoZGFFLr3+2KSxk3tCybvRnU1JA/a1DxOOG6yrKyIU13lwdyfecGZvcYA7V89hcUXu2HxYXXEOLrs2re8RQghxbkmJvIvU7lbvtMfG9PQJERM45RAOi8rfXt/A97dnBhAlOVYiCYNYUqeqwEGfP56xQtcE9rb7ON4XpHUotX36sdd7uHpBacb9DBMiCZ2qAicVeQ7uuayaupLsu52EEEIIIc6FX+3umvbYxCkLXLOtjS1x23j3umoe2NGaMbYi344nFMc0oarATstQBMNMxUCj4rrJrhYvXd7IWDm9xw/2sLAiM3EUSegYJtQWO5lf5uaey+ZSmGOb9vcIIYQQQszU88cGpzXOJHMNs5JlVXNDaQ7ragv49d7utOOqAuW5NgZDCSyqQpHLMlZRZ+LcVSCW5MWTg+w4OURcN+kPxPnDgR7efXlme4ZATMdmUWksy2FVdT53ralGnWS3kxBCiPNLEkwXqe89d/KMry1wWogljbRdSlaLwveea2F4ZHWIwniAMBhKoJIqr9fmiU5632hCxx9NpP2czdNH+3nySD8Aff4Yf3ldwxl/ixBCCCHEdOxqO/OScqW5NgYC8bRJDk2BH+5oHYunRsvAmKRK4o3+3DQYmfS+ff4ovgm9mqLx7Ct5H97bxa4WDwCGaVLsluSSEEIIIc4f0zTxTNI2YTqqChy0Tpg/UgBD1/n13m4SIytubBrE9dQCnB5/HKsKkYRJ81D2eScFONwTSCsbXOy2Zx370J5OXutM7V4vcNpw2mT3khBCXChSIu8idLBrGH9seuXxJlKB4hwLcd3IKIEXiOpjySVITY5oExZ3ZJ/uALuWWl0CjKwwSV2kKnDHyuyNp0tybNhHmje57RasmvxjJoQQQojz5+vbTpzRdVZNodhlxRtOZOz+7gvE0+KpU6u8TFL1hVy7RsFI7NQfiOOwpuIgh1Xlnsuqsl6T77SO/dllk/VfQgghhDi/3j3Nnt+nyrWrlOTY6BqOpR03gRZPdCy5BHBKW0sm6dRAsctCrt2CCfT6Y9hGJquKc2xZq+ZAerwkpfGEEOLCkv+CvQg9sP3Mdi8ZwFAofQWKRR0pmWeAVQGd8fItJW4bfYHs9WwBClwWfvKBdfzs5XYe3JMqO1NX7GJldT7VhS6uWlACgG6YfPe5JnqGo7xlUSlvWVzOn1w5jxMDQa6aXyIJJiGEEEKcV89M0hdyKgndZGjCDiNIxU76yE4lu0UhljTHdn0XuywMhSdf7Tu/1MXX3rmKL289xvYTQwCsqM6nriSHZXPyaChLlcfzheN87/lmwnGdey6r5o5Vc1LvEorxtkkW8AghhBBCnCvBxNRjsgnEDAKx9HkkizKeTHJZVcIjmSTDALdNIXia/uLXLyrlb66fz8cfOsCRkV6VVy8opdhtY3NjMQ5rKnnUPBjkf15sx6Ip3L+png9triPHpqEbJu9aP/fMPkYIIcQ5IQmmi1DbUGzqQdM0cXVt4pTf6YaZffmICiyqzOWTNy9CVRQuqy2g3x9DVRXuXltFTZErbfwfDvay7XCqJF5/IMY1C8u4YWk5N1B+zr5DCCGEEGI2jMZOChCdsPQ21UM6+wSJRYV1dUV8/MYFGKbJFfOK0Q0Tt93C+66oJc9hTRv/y92dvNiUKomX1E3+7a5l3HNZ9Xn4GiGEEEKIdKY584o5pzMaLikwllyC1E7xyR5ltyjcuLSCD19ZTyypc9WCYgpdFsrznbx/Yy2amt5P6ee7OtjTniqH7LJqfGTLAj6wqe6cfocQQogzIwmmi1D2zkbn3kAw+wpcTVP4tzuXoqoq//dKB7/a3YmmKXxwU11GcgnAZdNQldTOKLumos6wr2JCN3hgewuecJzbV1Sycm7BmXyOEEIIId6EdOPcTpKMOvWuhglD4exRWo7dwidvWQTA1588wTPHBih0WfnolvkZySUAp3W8lIvNMvOd3t5QnB/taEE34b0b5lJVkBmfCSGEEEJk0+UNnZP7jM4DjTo1dkroJpO07qaqwMmHr6xHN0w+/7sjHOzyU12YOnZqcgnANqEyjt0685J4J/oCPLinC5dN48NX1pNjl+lQIYQ4V+TfqBeZ27/94rTHWtXJa9gCNJbmcHJg5oFDQjf5yK9eZ3VNISf7AyQMk4RhcrDLz83LKjLGX7uwFE8oToc3wvWLSlGUmWWYHt7bxR8O9QEwGIzxjXeumvE7CyGEEOLNxzBN3vbd6fcQmCp2Wlju4lhfeMbvMRxJ8qnfHGTDvCKO9QUB8IYT7GsfZkV15sKZd62fS0I3CcaS3LM2e1+m0/nJS+08P1KCL6kbfOrWxTO+hxBCCCHefIaCUf7s569Pe7ymkNGncpRpQnmulb7AzOvttQ2F+eLjR1lTU8CJvlRpvE5vhD3tPm5cmjnv9OGr6nHZNCyayvs21Mz4ef/vpTZe7/QDkGPT+PBV82Z8DyGEENlJguki4gtP3g8pm9NNkEDql7NC+ioSTYWKPAddvuhpr20aDNM0GGZ9XSF2i4pFVVhTk591rKIo3L32zMu6TMxHtXsiHOwaZllV9mcJIYQQQox68nDvjMZPFTt1D2fGYnaLQoHTetq+lQCvd/k51O3nstoCun0Rit021tUVZh1r1VTu31w33dfOMHFh74EuPwOBKKW5jjO+nxBCCCHeHD7zyOEZjZ8suQSpuaZssZXbltptFIxPHnjpJrzU7OFgl5+F5bm83uWnrtjF+rqirOMLXTb+8rrGmbx65suOeLnFwx9tqpN+4UIIcY5Iguki4gnOLME0lWgy/Ze5Sqpp9emSS06LQnRCM2vdMPnmO1ditSiUnaeJi7tWV/HYgV68oQQJ3eSZowOSYBJCCCHElA62D53T+wWi6eWDVSCWNOk/TXIpz67ij6ViLt2EBeVu3nN5DUVuO/nOzPJ458J9G+ay/cQgsaRBKK7z9JEB7pUG10IIIYSYQqf39IuNZ8oTSt+9pCqpxJJ2msI2BQ4NXzRVOy8YS/Ln184jEjeoKnDitM28/N103Lqigte7UjuY+gNx9rR52TCv+Lw8Swgh3mwkXX8R+ZtfTX+b8pmYYtEuAO/bWEt5nn3s50WVbqoKnVg1lUj87LpDtQyG+Mq243x/ezMJffxtLJrKZTWpFb6aArXFs9tHYDiSwDvD3WNCCCGEuPCea/Kf1/uPRiun6/L0mdsW47CmQmoFqCnOoa4kh6RupMU7Z2J3q4cvbz3OL1/tSGvIXZhjZ2GFG0iVeVlQ7j6r58zUYDBGMJa9l6cQQgghLl7np3PluNGeTJPtfLKoCn9xbQOj+SebRcVtt1JfkkMonjzr3pp/ONjLl544xraRNgyjVs0tpLrQCUCZ205D6ezFToZp0uuPEp2sIZUQQlziZAeTGLOw3MVtK+eQNExeODFIQ6mbt6+p5mcvt/Poaz3kOy383fXzWTIn74zu/8ALLRwYWTHismrct7F27NxfXNtAY1kOuQ4LV84vBVK/hP/vlU4GglFuW1E57QBgKBgnx67hmEbjx6eP9PPjna0Ypsn7NtRy8/LMWr9CCCGEuPhcDP+Rft2iEhZW5PGWRWU0DYRYNTef9XWFfGXbCXY1D1Fd5OSzty2hKMc243vrhskPXmilZziKqkB5noNrFpaOnf/kzQvZdrifhlI3K6pTO7+jCZ3/eamNhG7y7vVzKZzGcw3TZDAYp9BlnVapmJ+93M5v93WT67TwN9c1sqoms8eUEEIIIS4+h7qGL+jzVeBDm2tYW1fEhnlFBKIJrllYhsOq8unfHuJ4f4Clc/L57K2LsJxB+bq2oRA/3tlKNGGwp93H8up8KvNTlXhcNo3P3rqYl5qHWFNTQGluamF1nz/Kr3Z34rJpvG9j7bRioYRu4A0nKM6xoalT9yD/2rYT7Dg5SHWRk8/cupjyPClrLIR4Y5EE00Xif19svmDPLnJa8ESStA1FeN+PXsEXSa1IXVyZh1VTebXVSyShE0no7GwaOuME08SVKIlTlrNoqsItyyvTjj32Wg+/eLUDgA5PhK+8Y8WUz/jhjhaeONhHUY6Nj22Zz/zy3NOOf7XVg3+kHM4rrR5JMAkhhBCXiD//6e4L9uzR2OnVVh/v/dGr+KNJbBaFaxeWEE0a7Gv3EtNNmgbCvHBigDtWVc34GYZpohvGyJ8hfkrp4xy7lTtXp9/3v3e28fjBVF+qYCzJJ25aOOUz/u0Px9jd5qW+yMXn37qEvCnK+u1p9xLTDWLBOC+3eCTBJIQQQlwi/uE3hy7IcxXAbVcJxAx+ubuLn77cSSiuk++0UJlv5/XOYQ52pxYj72v30e6JMK80Z8bPSejG2LyTbhgkT9lJXlng4K416bHTfz3fzO42HwCaqvJHV9RyOqFYks89epimwRCrqwv41K2LTptkiicN9ncOo5vQNhRh58mhjHcQQohLnZTIu0j8396ZNak+V/KdFkb3JkeTJt5IEpPUtummwRAAdSMl65wWlYpcO8HomZVEed/GWi6rLeCaBSW8c131lOOD8fHnxJLTKzGzp81HLGnQMxxlZ5NnyvELynOxagoWNfVnIYQQQlwaBsIXZgdTZZ4dz8hinEA0ObZQJZ40OdYXxGnVqC1OTYoU51gpz3Oc0W4rq6by3strWFOTz1tXVHL9krIpr4kmx58TS079TF84wb52H0nd5MRAiJ0np+5p1Tgy4eOyaSydI7GTEEIIIbIbTbtUFzgIjPSr9IaThEbaLwxHkhzo8rOg3D1Wvq622ImRJTk0HY1ludy7bi5ravJ51/q5zC2auv3CxLmmyDTitV0tXo71BUnqJvs6vAwGY6cdb9UU6ktS71HitrFqrizMEUK88cgOpouAYZxdff6JFKZXU/e6haVcvbCEYCzJD19oRVXAqkJs5PepTVNYV5vqi/TXb2lkdU0Bu1qG+PGLrfxmfzcf3bJgxjuZls7JY+mcJdMef9fqKrp9UTyhOLedsrtpMvXFLjq9EXLtGsumMelx15oq5pe7SeoGq0f6QAkhhBDi4jYUjJyze003dvrgFbXUFrto94T5xaudxBI6FlUhNrIr223X2DCvCEVR+Oxti3nhxADPHx/ki48fo7bYxT+9deal8q5dVMa1i6ZOLI169/q5hGJJErrJu9bXTDk+35nqeXC0N0B5np1VNflTXvNn1zSwpraQUredxrLZ7f0khBBCiDPz6N7Oc3av6cZOX7xzKUnD5OXmIfqDcZK6gQIkRy6uyHOwubGYfKeNf37rYl5q9vDkkX4+8uABVlbn87nbF8+4VN49l1UDUy9oHvWu9XNRd3fismrce9nU162oymVOvoPu4Sj1JTlTxnaKovCZkdJ8C8rdVOY7p/1uQghxqZAE00XgsQPnbvfSdNsh1ha7WFGVz+cePYw3nABgw7wS8pxWHFaVW5dXUuxO/aLUVIVrFpby4J5OkgYMjJREOdNSedPlsGp8dMuCGV3z91sWsL5+kLmFThqmOemxvGrqyRQhhBBCXDw++8iRc3av6cROClBd5GR5VR7fe76Z8MjK2y2LS1FQKcu1c9OycnLsqdDaadO4fF4xP9rRhgm0DoV5qWmIW1dMb8HMmSrPc/DpWxdPe7ymKnz+9sW83DzEsqr8afUEUBWFjfOKz+Y1hRBCCDHLHnix/ZzdazqxU45NZU6Bk1A8ydbD/SRHFuTcu64afyRBQ1kOV80vHeudXZLroMBlo2UwDMBrncP0+aNUFU69C+lsLK/Kn9GcUEmug3952xJe6xzm8vqiafVssllUrl5QOuU4IYS4VEmC6SKwt3XqciTnikWBuy+rZm+bl9/u78aqjdeKLcuz876Nk9ebbSxz0+6JkOewsLLq/CaXztRoMkwIIYQQb1z9w9FZe1auXeOtK+fwv7s6+M6zzZgj0yqaAksq87hmYfYdRm67hXmlORzs9lORb2dN7cVZEiXHbuEti8sv9GsIIYQQ4jwxzekuRT43qgvsXLWglE/+5iCJpIHdopLUddx2jSsbS6gpzp40WjYnj7mFTjq8EeaXuymbxsKXC6E018H1iy/OdxNCiAtBOd0vmoGBwOz+FnqT8YbjPLy3i9/u75nV55bmWBgIpfoF2FS4dnE5JTk23nFZ9WmbExqmyZ42LxV5jmnVshVCCDF7SktzJ/8XuJg1EjudX82DIbYe7OXxg32z+tzKPBs9/jgA5bk21tUXUV3gnHJHUjShs6/dx4JyN8Vu+2y8qhBCiGmQuOniIbHT+bWnzctjr3Wxu90/a89UgDn5drqGU/2JGktdLKrIY3l1Plc0nH4XtC8S50h3gBXV+WM7w4UQQlx4p4ud5N/WF9C3njnJq62+WX/uQCiJ3aISSxpUFbr4k6vqp7WtV1UU1tUVzcIbCiGEEEKkM0yTbzx1Yqx0ymzq8cexqJA0oKYohz+5at60rnNYNTZOMZEihBBCCHE+DARifOOpk/giiVl9rgn0jCSXABZV5k07dipw2iR2EkKIS4wkmC6gQCR5wZ5927IK3E4LV84vmVZySQghhBDiQtINk+ELFDs5LSr3rKtGVRVuWVZxQd5BCCGEEGImPOE4w7OcXBpV4rZy15pqFAVuXCqxkxBCvJFJgukCumPVHI5uPT7rz71lWTnv31SLoqTvbDNMk05vhBK3HZdNm/X3EkIIIYSYjFVTuWlJCT9/tXt2n6sq/Nk187h2UWavpYRu0DMcpTLfIQt2hBBCCHFRWVDmZk11Lns6A7P63DyHhb/fspAlczJ7d4fjOoPBGNWFTlRFKlUKIcQbgSSYLqAvXYDkkqrAZbUFGcklgK9uO8ELJwaZU+DgM7ctoqog1WdpT5uXSFxnU2Nx1uvOF9M0ef7EIPGEwfVLyiT4EEIIId7kZju5BKnYaeXc/Izj8aTBZx45xOGeAIsrcvnCHUtwWDUM0+SFE4PkO6ysqimY1XdN6gZbD/VR5LaxcZ6UlxFCCCHezBRFmfXkEoCmKiysyM043jsc5QuPHaHTG2FTYzGfuGkhALGkznPHBmgsddNQ5p7Vdw1EEzx1pJ8F5W6WzsmM94QQQkxNEkwX0Gx0sqwusHPd4jL+56UOAAwTDvUEWVefPumQ1A0OdA1jAl2+KN99thmLqmC3auxu85LUTe7oq+T+zfWTPks3TB57vQe7ReXGpeVnnYz65e5OfvFKB4YJ7Z4wH7py8mcLIYQQQpwLK+bkUuy28ezxIQBiusmuZi83L08v79I0GORwT2rS5khvgP944hiGCWCyt30Yu0XhQ5vruek0JfWC0QSPH+ilusjJFQ0lZ/3u33qmiWeODWDVFD58ZR03L6s863sKIYQQ4tLU4QnNynNuXFrK0R4/bZ5U3yVvOMFAMEZFniNt3ItNQ3R4IwDs7/Dyb48fJWmYBGIJjvQEyXda+IebFrKsavJET7cvwvYTgyyfk8fS04ybri8+foyD3X5y7Rr/eMsilp+DewohxJuNJJgukH5/dFaeY9FUHt6bvtq33x8loRtppVwsmkpDmZvdrV7yHBoHu/0YJmgq6EZqTMvQ6Ztqf//5Zv5wqA8FGAzGee+GmrN69/ah8MhEDXT6Imd1LyGEEEJc2p481DsrzzEVhR0nPWM/q8BAMIppmmmLZ+qKcphX4qJ5MEyh08LuNh8ANksqvoolTU70B7npNM/60tbj7OsYxmZRSOomVy0oPat37xqJlxK6Scvg6eM2IYQQQryx/edTs1M1xxNM0D6SXIJUeeGDXcMZCaa1tQX8/kAv/YEYdovGi82peMuqpeKr4UiSg93+SRNM8aTBvz5+lHZPhAKnlS/csYT6kpwzfm/dMOkeiZ0CMZ1jvQFJMAkhxBmQBNMF8viB2Zkk6R6OEk+m75V64eQQxW4792+uSzv+yZsXsrfdRzyp842nTxJPmtg1FYdTI6mbbKgvyvoMwzT59rNNvHB8EEjtzOodPvuE0DULSzk5EEI3DK6ef/areoUQQghx6frus82z8pwjPX6SxvjPBvDrvd1U5ju5YUn52HGnTeNf37aUA11+Orxhfvpyard4rl0jrik4bRpXNGQvUxdN6Hxp63EOdPkBiCdNWofCXHWW7371ghIGgnFybBrXLjy7ZJUQQgghLm1H+2dnoe7edl9ahZ6EYfLA9lbqi3PSSt7VFufw73cto2kgyJ42H08c6gOgyGVlOJqkPNcx6dxPpzfMN546SYcn9U2+SIKmgeBZJZg0VWFTYwnbTwxSnmvnLVn6bQohhJiaJJgukF/vO/89BCwqackli8rYhEk4nswYb9VULh9JInlCCY73BVlfV8i6+iIMw8TtGP/HZeuhXjq9UW5eVs5gMMZTh/vHAorKfBtblpZn3H+m1tcXsba2EMM0pXG2EEII8SaXGbmcO6qSKiOsKaQll2wWlXjSwDBT5V5O5XZY2dhQzOVmEZG4QX8gyi3LK6kpcmLVVBxWDUitkH14XxfhmM7da+fwh4P9vNrqHXt2Y2kON56D2On2lXO4aVkFqqKgqdK7UgghhBDnx+j8kgLoE7JLmpL6OZLQ8YbjGdeV5topzbWzoiofFIgnDN61vhq7VcNtt4zN/YRjSR7c24XLqnHXmioe2d/Dsb7gyLMVVs3N56r5Z7+Y5o+vquf9V9Rg09RZ7TkuhBBvJJJgugAi8cwJinPtw1fW8fNXOkjG9LFjZbl2inPs2C0q91xWDcDBrmFeafGycm4+a2sLx8besWrOpPd+7lg///V8M0kjtcr3o1vmk+e0MBxJUuSy8qW3r6DAZTsn36GpChryS14IIYR4M9v2esd5vb/DonDv+rn8eGd72vH5pS4MU6HEbeOOVal+RjtPDnKsL8g1C0qZV5paNasqCu+/onbS+/98Vzu/2tMFQK8/ypXzS7BpCnHdZF5JDl95x4pzNqkhi3KEEEIIcd8PXjyv96/Ms7O8uoDHD/alHV9enU80odNY6mZtbSGmafL7A70MBmLcsWoOhTmpuSKX3cJfXNMw6f2//VwTL5xI9cMMx3WK3eNzTJfVFfKpWxads2+xW7Rzdi8hhHgzkgTTLDNNk3t+8Op5f86cAjvFOTZCsfEt0SVuOyVuO3esnkN5noNANMFXnzzBYDDO00f7+dLbl1FV6Mp433Bcx2nTUEcmPvqGY2Ore4ejCSrynXzk+gXs7/Cxrq7gnCWXhBBCCCFM0+Rb289vgklRFFZVF5Dr6CIQTS3OsSrgsFoodtu4b0MNdovGax0+vvn0SSIJgz2tXr5570ospyR0DNMkEtdx2bSxpNHE3U/+SIIrGoqJXNtA22CYLUvLZMWsEEIIIc4Z0zTxZW4eOrfPUFRWVOWz7XDf2PxQnl3FYVGpLXbx/o21KIrC717r4YEXWjCB5sEwX7hjSca9dMMkljRw2cYTPf7I+N51bzjOX7+lEbtFxR9JcvfayRdECyGEmH2SYJpFD+3p5IlTVnecL//62DHyXFbKcm34wglUReH1kTr/ff4o//725fgjCbyhVNThjybp9UfTEkymafK1J0/wapuX2iIXn71tMTl2C7etrORYX5CBYGysnMua2gLW1BbMyrcJIYQQ4o3PNE2++1wzu1s95/1ZobjBRx86QIHTit2t4g0nsGgqe9p9QKqp9N9vWQDWe2AAACAASURBVECXL0IkkZpF8UUSxHUjLcEUTej882NHaB4MsaI6//+zd9/RcVXX4se/d7pGGvXeZUsuknvBHSdgDAYM2KEFSKFDAiGFkEZIfoSUl5f3SAESUiAB8hJCx1QDNqbY2Bh3y1W99xmNppf7+2PkkcaSLMmW5LY/a7Ey5ZZzjVdmc/Y5e/O9iyaiURSumJlJvc2Fxx/k8pmhSRGp8y+EEEKIkeQLBPnlmweoaLGP+r1qO1z89r3DJEbr8flVOt1+fEGFTypCJYCjdFqun59Ls90dbqdgc/XNetVbXfzqzQO0Obwsm5zKjYvyAbhiRiYObwCjTuGKGZloFIVVM7NG/bmEEEIMnySYxtA7pc002T1jci+/GuqjZNBp8AZU6NVy0dddIDczPoqLp6azs9bGhDQLM3MTIq5hc/nYXNGByxegtMHOBwdbWTE1nWijjgdWTh6T5xBCCCHE2anN4WXd/ubuOGb0eQMqzV1eDFqFQBBcwZ5mTL5A6PUFxWnsqu2ktsPJgvFJmA2RofTmivbwgp7N5e00d3pIjzORm2jmV6unjslzCCGEEOLstOFgS7jH41hw+4O47T3xkssX6PVd6PXl0zMpb3XQ6fJz2fSMPtdYf6CFijYnABvL2sIJpjn5CczJT+hzvBBCiFOPJJjGUGyUjnrbGN/TpKOty4tBp5CbaCbGqOOK7v5KiqJw27njBjw3xqgjO8HEoWYHydF6ijMtYzVsIYQQQpzlYow6LCYtbQ7/4AePoFiTnlaHF4tJS068mRiTjuvn5QKh/kbfXzFxwHMnpVtItRhptnvITTKTEK0fq2ELIYQQ4iw3PiUGnRJacDyWEs162p0+kmMMpMcaSYw2cO3cUN/vZIuRn18xZcBzJ6VZiDFq6fIEyEqIGqshCyGEGEGKqg78y9PSYh/jn6Uz1wMv72F7beeY3nNqZgz3XjiRfQ12Yk16pmbHDfsaHU4vHx5qZWpmHAXdjawH4g8EeeLjStodPlZOz6AkM/Z4hy6EEGKYUlIs0sTlFCCx08gIBIPc+c/tNNjGZuf3EZdNS+f6+blsKmtjXEoMBcnHjn36U9XmYGetjYXjkki2GI95rM3p44mNlQSCKjfMyyU9znS8QxdCCDEMEjedOiR2Ghl2t4/bn96G3RMY/OARogC3nVvA4sIkPq3sYEZOHCmW4ccye+tsVLQ5WDY5DZNee8xjK1odPPdZLVF6LbcuKRj0eCGEECPjWLGT7GAaAx8fbh215JJWgYEqxywsTCEx2siiwtDkxoeHWthaZWVKRiwXdPdOGkyC2cBl04fWQPGlHfWs2dUIhPo8PXzN9CGdJ4QQQgjR218+qBi15JJJp+DuZ2mvQQPnFCRiNug4f3IoTnplRz3lrQ7OLUpmdt7QyrTkJUWTlzS0xNRTn1Sxbn8LECrD94MVk4b4FEIIIYQQPR54Ze+oJZfio7RYXX2vHRel49yiZGKj9FxQnEYgqPKPjVV0OL2smpk55HioJCuOkqyhLYh+4uNKdtSESgOZDVpuXlww9AcRQggxKiTBNAZe390watc+VluCBps7/LrD4eVPGyrodPvZWNbGuNRoxqfEDOteTo+fv3xUgdMb4Jo52Yw76vzeu+FGcwnS+weaeXlHA3FRer55fiEJ0YZRvJsQQgghxpKqqmwZxf4BngHqxniDUNPuYnpOPACflLXxj41V+IIqpfWdPHb9TPRazbDuVW918swnNWi1CjctyifBHBmzBNX+X4+0/3xaw8dl7WQnmPjWsiJ0w3wOIYQQQpy63L4AVe2uUbt+f8klAJvLT4fTR2xUqCTwC5/V8vy2OgCa7B5+uWrg0ngD2VVrZc2uRpKiDdyyOL9PzNK7CFOvdpkjSlVVHl1fxqFmB9OyYyWJJYQQg5AE0xjYXWcf83vGGLWsnNbTQNHtD+DxhX59ff4gzuNY2fJ/W2p4d19ola3TE+BnV5REfL9qZhatdi9tDi8r+2neeLQPD7VS2+HkkmkZxJp6ehTsa+hk/YEWCpKjWTElvc95L++op6wl1ATypR113LRIfuyFEEKIM0Wz3UNLl3fUrj9QHifVYuCCktTw+y5vAF931sfjDxIIqgy3CsvfN1azqbwdAL1GwzfOL4z4/svzc/H5A/iDarjP00CCqsqbuxvx+INcNj0jYsLl48Nt7Ky1MjsvgXkFiRHnOb0BXtnZQKfbT3mrgylZcf3GV0IIIYQ4Pb2xpxHfsVYfj5Ki1BhyE3v6Jjl9PRkfj+/4sj9PbqzicLMDgMRoPVfPyYn4/isL83huay1RBi3Xzcvp7xI9Y/AHeGVHA/FmPRdMTkVReqo7vbazgep2J8tLUilMjew3vr3GytrSZlSgut3BRSVpZCWYj+t5hBDibCAJplH26LrSMb2fAqTGGvnKgryIOv4ZcVFcPTebHTVWJqVbmJodh9cfZFN5GxPSYsiIO/Fminqthq99fvyQjn1/fwt/WH8Yb0CltMHOzy4PJasCQZXfryujtsOFXquQaDYwb1zkRInFqA8/a1L0sXsbCCGEEOL0cvtT28b0floFshOjuOe8Qoy6ngzSeZNSONRsp6bdxZLCZEx6LTaXl+3VNmbkxBFvPvEd1AnRBu69cOKQjv2/zTU8u7UWgDqri7vPCyWr6jqcPLL+MF2eABvL2ii6ZjqJMT3xkV6rEBulp9Ptx6TXkCF9noQQQogzypMfV43p/QxaGJ9i4f5LJkUkba6anUWz3U2ny8+qmaFWCw02FwebulgwLgmD7sR3UBelxvDDi4dWUvh37x3mw0NtKECX28/qWVkAbDjYwl8/qiCgwv4mO7+7ZnrEc6RYjMQYtdg9AeKi9Fh6LYgWQgjRlySYRtlbpdYxu5degWduPQejTotW07fv1tVzsrl6Tnb4/c/f2M+2aitJ0Qbuv2Rin1UbR7vunBwcXj8ub4Cr54auc6jJznv7WyhINnNhydBXw1Z3OPF2r7Bp7erpseAPBnF4/AD4Aiptjr4rmO85v5CXd9STHGPksiHslBJCCCHE6cHpcjN2rakhOUbHn26YjV6rQaNExk4aReHOpT0LZ5xeP/e/XEplm5O8RDO/vnIKZsOxQ+mvLsxFp1HQaTV8eUFoh9KWina2VVuZlRvPOUftNjqWxs6e0se9Y6cOlx+nN/Sn5vD4sXv8RyWYNHx7WSHvH2ihMM3CjO4SgEIIIYQ4/W0+3DKm95uTG8sPLylGp1EikjIA0UYd9/VaOFPW4uCh1/fR2uVlRk5zeGHxsdy4MI/XdjWSFGNg9cxQQujtvY1UtDq5oDh1WK0eWrt3xKtAva2nhGCbwxtuN+Hw+FEJLWA+IifBzD3LithVa+Oc/IRwCUAhhBD9kwTTKNqwv3FM7zcjN37QiY4jfIEg5S1dQOjHdWdN56AJJrNRxz3nF4XfB1WV3713mKr20G6jWJOeBeOThnT/FVPS2FPXSYfTy/LitPDnRp2WL8zK5INDbWTHR7G8OLXPuckWI7cskbJ4QgghxJnmhr+N7e6leQXJEbuWjqWi1UllW6hEb1W7k/KWULm5Y8mMN3PfRT0TLU2dbn6/7jA2l58PD7Xy8NXTSI0d2o6ii0rSqG534guoXNRrUU9JhoWV0zLY12hnalZcvw21i9IsFKUdO84TQgghxOnnobcOjen9ZuUlDrkn5Y4aazjJU9HqxBcIDnrutOx4pmX3LIbZVNbG4x9U4Auo7Knr5A9fnN4nsTWQC4vTsLl8mA1aLu5VHnjltAzKmrto6vRw/qTUPouMAOYVJPYpOyyEEKJ/kmAaRS9ubxiT+0zNjCExxsQXz8ke/OBueq2GGTnxfHS4lcz4KBYVDi0x1FsgqNLV3cvJF1CH1S8hxWLi11dO7fe7y2dkcfmMrAHPVVWVQ81dJMcYSJQSeUIIIcQZwzcG99AoMDsnjoQYI1+af+y+R70VpcYwJSuWffWdTM6IZcJxJGysLh92d2intsPjx+bykxo7tHOnZMXx+2tn9PlcUZRBF974AkEONXWRn2we8mIkIYQQQggAg1Zhdm48GQlRXDJt6FVkFhcmsX5/M3VWFzNy4oacmOqtpcsT7i/l8PgJBFV02qElmM6fnMr5k/suWtZrNXx3kBLFTq+fylYnRWkxxzVuIYQ4m8h/YY6iH6yYwK3P7Bz1+1S0ubhn2QTSjrECdletlXdKm8mMj+LaudkoisK3Lyjihvm5xEXpMQ2ha7Wqhvoj7a2zUZIZy13nFbJqZibvH2whM87Eiilpg15jJPxhfRnvljaTYNZz30UTKckc4syMEEIIIU5pq6Ym89Lu1lG9R1CFGqubH1wy+ZgTBuv3N7OtxsqUjDgunJKGQafhZ5cV0+rwkhxtQDeEyQZfIMjPXttHi93DosIkrp+Xy8ppGeyt76QkM5bC1L67jUZaIKjy4Jp97Ki1kZsYxc8uLyEx+sT7RwkhhBDi5IvRQ9cor9DxBkKLi29cmH/M457/rJaqNidLJyQzJz+RtFgTv7lqGlaXjzTL0BYHW51eHlyzD18gyKXTM1kxJZ39jXYabG4+NyFlSPHXiepwevnxy6VUtTuZlhXLg5eX9NuGQgghRIgkmEZRenw0Ji24R7mZQJcnwO/eOcQPL5nEr946SFOnm89NTOH6eaFVuU9trOSF7fUEu2vM1nY4uGXJOBLMhn6TUs02Fz96pRS3L8CX5uWyvHsr8b+31PDuvmYAGjpbGJ8aw+UzMrl8RuaAY3N4/Pz5wwqcHj9Xzc5mQvqJl2cprbejAu1OH1srO/okmFQ19KBD3TYthBBCiFPDTUsnjHqCCaCx08OHB5vJT7HwyLrDOL0BrpqdxfmTQ4tl/t+re9labQNgY1kbTXYXXzwnF71WQ3o/sdO2ynZ+t64cjQZ+cNEEJqSHYpNfvrmf7TWh6zy3tY7545IG3W1UZ3Xy9KZqNBqFmxbmkzzECZmBWJ1e9jV0AlDd7uLTyvY+fTODqoqCxE5CCCHE6eZfty9k5SMbR/y6CqHeRUfsru/E5fWzs9bG059Uo9Uo3LakgClZcQSDQe7453YabKEekTtrrXxpfh7LJqdi0mtJ72dB80vbanlhWz3RJh2/+cIULFGhxS/3Pb+Hhu6+k09vquL8SSkRfZ36s7PGyqs7G0gwG7h9acEJ7zjaVtVBVXuoLHJpg522Lk+fksZBVe23tJ4QQpyNJME0ykYyuaQAK6aksqfORnWHJ+K7fU12fvveYXbWhiYx3ilt5tq5OQC8tKMnuQTwwaF26q0efn3l1PAPrz8Q5LfvHqamw0mX209zd7m7JzZWsXxKOqqqsrvOFnFPi2nwvz7/2lLDuv2hppNOb4Cfr5pyXM/e2+SMGOqtLhLMembnRTaq/rSynSc+qkQFvrIwjwXjhl/6TwghhBBnBotR4fMTU1l/oAl7ZOjEU5/UkJ1g5lCzA4C39jZz/uQ0Suts4eQSgNev8txn9dS0u/nRJZPCn1udXv733UM4PAEarC7s3WWD/3vtIf7y5dkEgiqVrY7w8YoSan49mKc2VrOxvB0Ag1bDN5cVDXLGscWbDUxMt7CrrpPchCjm5EX2E3htVwMvbq/DYtTzzWWFFCSP/q4qIYQQQpyasuONzMqN583dTfjUyO8efG0fHn+Q6nYXAG/uaWRKVhyPf1ARTi4BdDj9/H5dGdXtTm5e3LOwpqy5iz99UI5eq2F/Qye+INjcfn751gF+sWoqLp+f1i53+PigqqIZws6hv2+q5nBzqMd4coyBa8/JOZE/AmblJpCbGEV1u4tJGZY+O7//8mEFHx9uIzPexI8unki0UX9C9xNCiNOdJJhGUSAwsluXVOCNPc39fucPwuaKDnSa0OuEaD1ajYIvEOz3+Io2Bx1OL6mW0CqM3607xIZDbUAokXW0x94vZ3e9Pfw+1WJg6YSUQcfcexvxSK2K/cZ5hSwvTiPFYiQ5JnJV77v7mqm1hgKSd0ubJcEkhBBCnEbaulwjej27R+XVXU3938vhw+rsSSQlRocmB/xqv4ezv7ETVVVRFIWgqvL/1uzjcEsogXT0OtlAUA2VxutVs2ZShoWMuIHLGR/RO14aidhJq1H46WXFHGi0k58cTcxRSa539zXTYvfSYveytrSJ288dd8L3FEIIIcTY+NO6AyN6vVqrh1pr/7HTnno7hl79j1K7d1kPlAQ6sgAawOsPlQ1uc4Zio95nKIqC1enlgVdL8fWawlo2OWVIu4Q0Sk/wphmBCnoJ0QZ+/YWplLc6mJhmiSjL5/YF2HCwBZvLT5vDyzulzVwxc+Ae4kIIcTaQBNMouunJLWN+z1SLiQ6nl3aHlw8PtTI7L4HVs7J4p7QZi1FHg82FLwiBILy8vZ5r5mbz67cOsquuM+I6GbEGPH6VryzMY0tFe7g03hHFGUPre/TFc3Jw+QI4PH6unJ09Is+oKAqTB7h/77I1Q5nEEUIIIcSp46t/3z6m9wuokBFnpN3ho6rNyeFmO+OSo1lcmEhpvZ14s46qNhcBFawuP6/vbmRqViy/WXuQyraeZJhWA4lRerQaDfctn8Czn9bwWbU14l5Th9gz8qbFeRi0ChqNhq8syB2R59RrNUzJiuv3u7RYE2UtDvQahbxE84jcTwghhBBj4/XStjG9nzegkhJjwObysbuuE5vTy9Vzsilr7qKx00O8SUdluwsVqOtwsavWRiAY5JH1ZeHkEoBJr2DS67CYdNx34QT+++2DVLQ6I+5VmDq02OmWxQXdJfL0rB6hZE+0UcfUfmIng05DWqwRm8uPxaSjKC1mRO4nhBCnM+VIv5r+tLTYB/5SDGo06uAOxGLSoVXAFwjg8Ib+tcWadCiKQpRew62LC3hiYwV11p5tywvHJ9Hl8bOr1tbneitK0vja58cD8MOX9rC7OwGlAFOyLHxrWREpllMvgRNUVd7e04SKykVT0qUmrhDirJGSYpH/wzsFSOx0YsYydooz6dBoFJxuH57u1bJJ0Xo8viApsUZuW1LAr948gM3tD59z6bR0dlRbw7ule7t5UT5XzMxEVVVueeozmu2hcsM6TSjmuuf8Igy60W9MPVwef4DXdzWSbDFyblHyyR6OEEKMCYmbTh0SO52YsYqdDFoFs0GLRqPQ7uhJFKXEGHB4AxSlxrC8OJXfvnsoYhfSVbMzeWtPU7iU8BEa4IGVk5mdl0Cd1cXXntnOkdNMOg2XTEvnKwvyTsn+kG1dXt7b18TEdAvTc+IHP0EIIc4Ax4qdZAfTmFEBFQ1+ghi635/4D6UCZMWbqLe5I/osAXR2T4jYXPDYhjLaegUBZr2GS6em8ej75f1et6lXo4LYXr2WLuqVeDoiEFRZf6CF+Cgdc/Ij6/oPpMPh5ddvH6DV4eXC4rQR292kURRWTE0f/EAhhBBCnDQej4enn36SxsbG8Gdev5+GQ05SZ1+K1jB6i1j0GoVkiyGiV8ARR2KlrlYnv3zrQDiWglDy6fMTknlnb/9lY+qsoR1NiqIQbzbQbPeiAW5dUsDFUzMijnX7Ary3v5mi1BgmpFmGNO4DjXb+uKEcjz/I9efksHiEkkFGnZbVs6S0ixBCCHG6+d839o7JfaINGqKNuvDimd5auvt376y1cbDJHpFcmpAaTXqsqU9yCSAIVLQ6mJ2XQKxJR2KMgdYuLya9hl+sKqEoNTI+arW72VTeztz8RNKHWK1mw8EW/v1pLSadhjs/N27IMddgkmIMXD33xPo8CSHEmUQSTKPkxkd7VpEEfW7mHv5/zPXtQAN06ZN4svAvaPV6TvRfgQoYdZo+yaXe9Bpw+SJ7MTl9QbZUdnDepFTe3dcEKthdXlx+lbgoPRdNSQsfe/d540mPiyLaqO13u/EfN5Tx9t5mDDqFWxcXcNGUwRM8b+xpZE93T6e1pc0jlmASQgghxKmttraGm2/+Etu3b+v3e/vO18i76kGMSSNTHu5oGo2C03vsPplmvYZOlz/iM7vHz+FmB0snpLCz1opGo2B1ePEFVdJjTazoFf9854Ii3tzTRE6CieUlfeOin7+xnx01NuKidHz/ookDlq/r7Y3djZR193x6a2/TiCWYhBBCCHF6Wl/ejDHoxnj4bardsWRTQ5W2kLhxc9BFWRiphc0Gnabf5BLdV1cBi1GH3RMZOzXZPaRYjMzNi6e63YWigbYuD6qqUJQaw0UloXkni0nPdy4o4pPydqZlx/VJLjm9AX6yZh/V7S5e29XIf31hCvFmw6DjfntPE7UdoQVAb+5pHLEEkxBCiEiSYBolrSqAirutjq7nv8ZjTb1/jFtJSb2d7141hReS7jvhe+m0/QcMei2smplFeYsDu9tPVbsTraLg6J5U+fBQG75AkPQ4I/dfPBmLSdddJ9fJa7sbqWx1EBelp7HTwyVT00mL7X+VSFV76Afb61c53NI1pDHnJZkx6BS8fpWk6MEDAyGEEEKc/srLD3Pxxctob28nMzOLa6+9nqioKADqPvozG3Y1UtHcSNdf7yb/8ntJnLRkxMfg8QdJjzVgOyqBBKFdSvPyE2l1emju9NJgc6PVgNMbxOtXeW5bHR5/kInpFn60YiJBFTQK7Krt5LXdDRSlxGD3+PD4glw/LweTXtvnHkFVpaY7drK5/JQ2dA4pwZQWawpP4kjsJIQQQpzdqqoqmVf1GNs2vMc7laGY5kD3d4bYZIquvJ/07Ey6GFofo2PpcPpJi9HT1OXr892RXUruQICadjdtXR5QFDz+IDaXn9+9dxhvQGVxYRJf+9x43L4ABq3CugMtPLmxiulZsVS0u4gxarlxUT5aTd/5rQari+ru2Kne5qaspYvZeYNXz0mKCcVLCgw4nyWEEOLESYJpFAQCASBAYuXLfPzsX+j0QJYFluaHvt9QCXXNzTz013WMv3oJhoIFJ3S/A00ODAp4j9rF5AvAxsOt1Hb3XSpMMeP0BtAqKiganF4/Ll+QTref9/b37CJ67P0K9jWGdhcdmcg42NTFr1ZPAUBVVV7d2YDV5WPVzEwWjU+kweoiyqBlSWHPalqHx89j75fjDwS5dUk+yb16Ni3uPq6qzcnFUtJOCCGEOCs89dTfaW9vJzc3n7Vr15OYmBT64tOnSdQ24FgAN74CL+7zcOg/Pydr6Q1kL71hxMdR1d63hxKEYpeNZa1Y3QGMWshNNNPp9qOoPvR6La3dZWC2VnZQ2mBnWnYcLm+AR98vo9nu4T2lObyrvLnLw3cumACA1x/k+c9q0WkUVs/KYt64RDYcaCEt1sjSCSnh+zdYnfzlo0riTXpuWzouIkH1xXOyiTfr6PIEWDUzc8T/TIQQQghxevjxj7/P448/Fn6fYILl42GXZgoV9VbcbbWU/v1ezrl8HqVT72ckdjH1l1yCUIm8ynYnHr9KsllHTqKZTrcPq9OL2agPlx/eWNbOLYsLMOm17Gvo5M8bKnD5g6w/0IIvEAqePD6V6+aFSs+1O7y8uqOetDgTF0xOZW5+AnvqO5mQGsPUrJ6+R3vqbPx7Sw0T0mO4YX5eRB/uu88rJCfRjMWoHVKlHSGEEMdHEkyj4Io/vsf/8hDff38HnR5YkguvXQexxtD3nR5Y+S/4oAqcH/7xhBNM0De5dESdtae/wOEWJwBaBX6+ajJPb6phb0MnsSYdk9N7tgq7fD1lY45c1ubs2YH14rZ6/rGpChWoaXdy/yWTWV6chk6jiWhe/aOX9lLWGirlUtHm4M9fmh0xtsWFySwuPJ6nFUIIIcTp5u233+Txxx8F4M477+pJLgFJW76ORgnFSs9dCQ9/Ave+A3UbnmH6/AW0GccPdNkR5faruP2hOMgTgEPdsVOUXsNPL53Mf689RJ3VTWacibyk0M4rXyCI0xtaOdy7ZLHN2TMR86cPynmntDn0ucvPnUvHccO8HKL0WnRaTfg6972wF6srdF6rw8uDl5eEr6EoSp9eTkIIIYQ4uzQ01IeTS1NSYVIy/O4iyLSAqu6hNpDIyjemsXP7Ltas+4wJU30EGL2dz+294p1Wp59WZygmSonW843zC/nNO4ewufxkxZvQd1ffsbp8uP2hNg6BgNrrWj3zTg+/c4gdtTY0Cqgq3H/JJLrcfmJMunASqbbdyU9eLcUbUNlZ10kgqHLjooLwNQw6DVfPkXYMQggx2jSDHyKGaz1XUrZrBx9Wh97/fkVPcglCr395fuh1gq+ZEkpHbSwqYD6qPEtAhT9uKGd8ipn4KB1Z8SY+Lmvj2U9rCARVrpmTTUmGhXn58UzPjiM/ycxlM3pWyrZ0ecKJp47uYMJs0EUklwBau3qSWzanD1U9RqMoIYQQQpyxVFXlG9+4g0AgwMqVV/DFL/baleTzRqyr1WjgOwshIyb03v/WTwn6PYwlizEydnL5gjy1qZrs+Cjio3Rkx5v495Za3trTSGyUnqtmZ1OcYWHR+ESKMywUpcZwRa/Yydpr8qWje/LEYtKHk0sAXR4/dnfPcY22/ndZCSGEEOLs5XKFSsXlxcHuO+G5q0LJJQBFgRxdO2vP3wWA1Wcc1eRSb7HGyPmgFoeP9w+GdmvHm/Wkxxp5/IMKNle0M78gkUumpVOcYeHcCcmMT4lmSmYsV8zoWUhzZMFNUIWmTjcaRSE2Sh+xQ6ms1YG3V4Kq3iqxkxBCnAyyg2mEBR5dyrffgD9tDb2/eSZMT+t73JGfRBVIp469FI/amLLijeEVuEc0d7p5z+7F4Q1gdXWxrzHUOykQhBk5cXj8QRzeAF9ekMvc/MjatpfPyKCqzYnD6+ey6QOvpL14ahrPb6sHYHlxGopy4tuyhRBCCHF66ujoAOCvf/1HREwQ/+fkfo9/+EK47kVYu7OFcakvkbLg2jEZJ0BuQhR7GyP7Sla3O2h1hFblbqmyAqDXKMQYdcSb9djdfoIq3H3eeHITzRHnXj49A6vTh0bDgLFTfJSeeeMS2VLRjk6j4aq5suJWCCGEEP3TnWLLxdNiTXQeNe+0v8FOnS20SOj91v8/KgAAIABJREFUA60EgQ8OtfI/V04l1qTH6vRh0mv4xRUlmI2R05Mrp2ewZmcDcVF6Lp/Rf2ngheOTGJ9ST2WrA7NBy2opISyEECeFJJhG2IOvb+dPW8GghT+sgFtnhVaRDEQBiqjivREcQ1K0HrvLjzeoogE02r4D8PhVNATCYwiXwnN5eX13A4dbQqXt3tjd2CfBlBEXxS+7+zH5A0HaujwkRBvQKAqBoMqTH1fSYHOzrDiV5++YH7HCRAghhBBnt4gFJ34vWvrvDHDNFNjWCL/+GLo+e4HM2RfiM8QPcPSJSYsx0OLwElTBpFOwewN9jrE6/WiAIKBRQitqfUGV5i4Pm8raqekIrSh+fVcjd35uXMS503Pi+d+cUL8Aty9Ah8NLQnRoRbHD4+fPH1bg8ga4Zk4237toosROQgghhOjX+/9+fMjH+p02nM0VxKRmE0Q/ouNIjtbT2t1fKdakpdPt73NMs71nB3qw+38dHj+tDi9v722izeGl3ubmjb1NXDkrK+Lc5cVpLC9OC5/T6fIRGxV6hpp2J//cXI1Oq+EnKycTd9TOJiGEEGNLEkwj6KNHStga2rDDoxfDLbMGP2dvCyz1reMv+q8QGKEffAXwdjcBCAIHGh19jgmqPT/wWfEmzAYt0UYdq2dl8dquhohj27q8JEbrURSFBpuL368rw+Hxs3JKOm/va6ayzcHMnHh+cPEk3trTyCs7Q+fXWV3ML0hECCGEEKKPR5NIwnfMdNEtM+HRLdDcbifzmRtJufFJOpWEER+KVqsJ90/y+FWq2119jvH3qvQ7PiUaVYX0WBOXTs1gb50t/J1Bp2Bz+YjrngTZWWPlqU3VKAqsmJLGc5/V0ebwcmFJGrcsLuD/tlSzbn8LAC5vgJ9dUYIQQgghxNFW3DGdz16sAOALkwc+LsUMi3PgoxrY+7dvcv6tP6YzeTYjuUjHbNRCd4LJ7QvS6Q4tzjmyGAfA1/1Co8DkNAueQJBJ6RZKMizEGLW0OUI9wuOjdDg8fqK7dzGtLW0K715aXJjEv7bU4A+qXHdODpdMy+CpTdV8UtEOgEGr4RvnS3NvIYQ4mU6xTbWnr5WPbOQyasLvp/VTFq+3icmhXkx2L1z+pJXp/s0jNpYjq0j6k2Yx9AkpGjrd/GRlMQ9eXkJarAmLqSfvuKPGyu1Pb+PxD0JBzCs76tlT10lFq5N/b63lQFMXHr/K9hor9R1O9jd1hs+V9SNCCCGEOJrTGSqfciS5dKx4oSgJttwaer2j1k2J46NRGVN9r35HR3eMTO/dSLNbi93Dw9dM53srJmLQaYjpFTu9tquBO57Zxis76gB4ZWcDB5u7ONDUxXNb66izunH7gmyttNJid1Pd1qucjARPQgghhOjHykc2UnAgNC+zKAd+tWzgYxUF3v4STEiCoM9DavlzGBjZfpbV7T2xk79XHySVUFWd3oIqeAJBHr5mOrcvHYdGoyHKEOp3GVDhsfXlfP1fO9jcnTRas7OByjYnO2ttvLy9nnanj063n80V7VS0Omix99xbNi4JIcTJJwmmkyQxCj66MTSPsLsxSF7V86N2L32vf8tNdm+fiRO9RhNeKXKwyc5L2+vD3/mDoUBga1Wob0JKjCk892F19TTlzk008/t1Zbx/oA2zXsP07FhuWpwvfZeEEEIIAUBubj4Al1xyAa2bnh00uXREcQqkRoden8/7IzYezQA3P7pkX2Nn3wkZvbYnuNpY1saHh9rC7/1B6PIE2FwRip3izT2TLC1d7vC1M+MM3P9KKTtqO7EYtSwYl8hNi/KP82mEEEIIcaYr7N7E/Wk9vHHo2Mea9bC8u2LveKrQ0LeE3XANFDv1WmeDCrT1s+hZ26t1wys76jnYq9elL6jS1uXlk7JQPHVkFzhAU3cySQFiTFoeeGUvZa1OEs16zpuUwpfn5x3/AwkhhBgRkmAaQbdxb/i1enQWpx9T02D5+NBrV3B0qhUq9GxL7o8WWFSYSH2Hk/ue382PXtpLl6e/vgNeHlxTSoJZx82L8ylMicYbCAUPyTEG7l1eRFV3ORmnL8j8cYl9ejedDM12D85++igIIYQQYuwoisJTT/2LuLh49u7dzdOfVB/XddZwjOW6wxQcIFZT6buLqTeDFpZNSmFvvZW7/7Wd/377IL5A3zPKWxz811v7WVyYxLVzs8lOMIVjp3HJZs6bmEq9NTRp0uUJcMP8XAqSo0/4uU5EIKjSaHPj9R8jeBRCCCHESTF5yWJWTgBvAH64bujnafCj5cR/2weKnfppvxQhWq/houJU1u9v4uZ/bOVvH1X2O5rPqq38z9oDXH9ODqtnZpIcbeBISDI7L46seDNWV+hmOq3CN88vJM48sr2lhssXCNJocxMY6A9HCCHOApJgGkGpdHDkt+2+d8HmDv0A+4/xO3Pkq39z1aiMadCfOAXe29/K1/+1k32NdtwDTCh4/CqfVln528dVLJ2QzGUzMjF1b40qSjHz2/cO4wsE0ShQmBLNosLkkX2Q4/D4B+Xc8cw27vn3Dg412U/2cIQQQoizWnFxCXfddQ8Ajz76e3Y2Df8ad/MEQ4huTshg0y+BIPxrax3ff7GUyjYX/gEmFBzeAB8dbufvH1dx/bxcFo1PRiG0+nd8SjT/2lqLTqOgVWBWbjxZ8VEj/izDEVRVfvHGfu54Zhvfe2E3na6BSy4LIYQQYmytuWshV+g/4tcXhN53eQc/50iEspWZOLCM2tgGi53cviC/W1fO/75bRnM/VXWO6HD6eP9gG89vq+PGRfkUZ4bGbNJrSI+L4r3SZrQaBb1WYVZuwkmvmGN3+/jeC7u545ltPPT6PkkyCSHOWqOzbeYsdNPMGJ7YfhUPnPs3tv8HPqiC4sdgSS7My4JLz8mkSFsfcc7eZni3vPuN5uT8qziy4HaoP4P+oEogCJ+fmEJ8lJ4Gm4vDzV1sqrACYDHp+NkVxfj8Qd7Z10RDh4vSRjtZ8VF87XPj0Q60p3oUbK+24QuoNHZ62FjWTlHa6AVUQgghhBjc9dd/hZdeeoHS0j1c8W8o+0aoufNQ3cLDoze4Iepns9IxHUlAXT8vh5yEKFRV5b39LVR37/zOT4rix5dOptHmYk99J4ebHdS0O5mWHcd183JHevgD6nB42VljJaDC4RYHm8rbuLAkfczuL4QQQohjW85b/ENzEQDlHbChEpbmRx6jqvDMLnjzMDy7N/SZUfFxMhs9DremzJHY6VvLiijJjCUpxsDTn1TT4ghl1WbmxXPn0gIONHZS2eZiZ42VdqeXZZNSWVY8SEP0EbSlop1DzQ4AdtTYaHN4SbX07d0phBBnOkkwjZBVi6bxxPaNPFjwGm/cspKrn4ujvtnKs3tDP+rfe6+RHy0GXfeesRYn/HVbaIeTOa2A2LypJ/cBhiDRrOcLs7NIijEAMDM3ntpdTtYfaAkfkxlnYluVtXtHU88MzN56O0WpMVw0ZewmKsalmKmzuog16ZiWHTdm9xVCCCFE/5KTk3njjXfJz0+nygof18LinKFvqVfRo5zECZLhykkw8aUFoSSRoigsnZjC4xvK2VFrCx+TnWDmhc9qeWZzTcSCnwPNXSwpSiYn0TwmY42L0lOQEsP+RjsZcSZm5MSPyX2FEEIIMTRr7lrIpX94hYuLLueNQ3D+UzAtDb4wueeYTbXweq/+TFmxUDXpTrRjP9zjUpJp4cvzQ7GTTqvh4qkZPLimlKo2V/iY/KRoHn73EO8fbIs4t7XLy3mTU9GM0c6madlxZMaZqLe5GZ8STcJJLtcnhBAniySYRtCauxZy1SMb+U7Sm0y8uYGvlf8CrLX8z0c+Ohx+frqh7zlJUz9PwSX3oNEZxn7AwxRQVS6bntnzPqjy5u7GcE3c5GgdP7pkEk9+VNmnF4FOA7FRY/tj+50LJrBofBuZ8VEnvaeBEEIIIULMZjMXX7ySN95YwwXPGHj4Ai93zB64cXRvfgycTv/prtFoWDAuKfze7QvwweGehTlFKWa+fUER97+0p89u8ii9lijD2E0H6bQa/t/KyXxa1UFxhoUUi2nM7i2EEEKIoXnt7s/jUo08tt7DH7bA9sbQP71p9EYyF11DbHIKMQVzUaJOn0UjCVF6xqfGhN+32N3sqLGG388rSOBL83P4yhNb+5xr0mvGdBlSisXEL1aVsLuuk7n5Cei10oVECHF2kgTTCHvuroWsfGQjTn0Gayb+gcW8xQMl5RQ2vcbPqhagoJBCG4cpwJ8+k8TJi0963dihyk6I7A3w2PrD1HS4w+9dviBr9zYxId3CB4daCaig18KUzDjm5CWwcHzS0Zc8pg6nl9d3NZKdEMXnJqYMe7xajXJK9IISQgghRKQ///lJHnjgBzzxxF/4+uvwSS388RLovfDzSHTkD0J39ZHTTnF6ZHnen7yyl05XT6GYTrePt/c2UZRmobSxC4Bog5YZOXEsKkwmOWZ4ZVYqWh18dLiVqZlxzMgd/mSS2ahj6YThx1xCCCGEGDuHV+3nLs0s7jqng//bDUemZRyYeFVzGXFTLiAqOefkDvI4zSnoiV+cXj/3vbAHX68mT11uHxvL2pmQZmFLVQcACVE6ZubFc1FJ+rDn13ZUW9ldb2NJYTL5x7EwOSnGeFzzVUIIcSZRVHXgQvItLXbpUHeculxe9tR1Misvnjue2U6L4/RplGzQgveoIrnRBoXfXDWdHdU2shJMzMxN4KtPfEqbs+9zTcm0oNVqiI/ScduSAmKjjm931o9e2sOuuk70WoVvnFd4zB/tDoeXF7bVEWfWs3pm1pj2ehJCiFNBSopF/o/vFCCx0/A8//yz3HvvPTidTiZPnMDTv7iH3MU3kPjnLDT+Lhq74Lrn4f0qiDIZmfKtZ9HoT72dNQp9+1kWJEXx7WVF7KyzMSUrnrzEKK760yf4+/kbMjsnDncgyISUGK6bn4tJP/ydS25fgG8+u5M6q5tYk46Hrig55g7ushYH7+1roiA5mgvGsF+BEEKcCiRuOnVI7HT8qtscdHn8ZMcZ+crft+EPz++dnn+9r56dwdz8RA40dbFgfBItdg/ff3Fvn+P0GpiVl0Cny8fiwiRWTM04rt1DZc1dPPBqKZ1uP9kJUfz2mmkYdQPHYFsr29lWZWVWXjxz8hOHfT8hhDidHSt2kh1MoyQmysD87t0zT9w4FwCf38/qP205mcMakjijlhZnZIbJ6VW58587ADBqYcWU9H6TSwAHGrvwBVX0WoXzJqUyK3fwBFN5q4NH1h3G6Q1w9ZxszpuUSlt3A0dfQKW2w3XM83+/7jBbq0LbptUgXD03e9B7CiGEEOLkuvLKaygpmcqNN17PvgMHmfOFr2M0frv7WyMejwcAfXQC+Vd+/5RMLgHEGsHmifysos3F3c/uAiDBXM/M7Lh+k0sA22tsBIE6q5ur5+YwlDqAm8rbeOaTanQahdvOHUdarJFWe2gQnW4/Fa3OARNMQVXlt+8eorLNiUGrYDHpmD9ueDvNhRBCCHFy5Sb1/M6/9PWFAKx8ZOPJGs6waIDgUZ/957MG/vNZAwBr9zYRpet/LtMXhM0Vod1LvoDKZTOyhnTPl7bX8eaeJhLMeu67cAKVbQ463X4AWu0eHJ7AgAmm5k43D797mE63nw8Ot/Lw1dNJsQxvp7kQQpyppEDoGNLrdKy5a+HJHsagjk4uQeSqXE8AXt7Z2OcYAK0COm0oCAgEVTy+o0OG/r25u5FDzQ7qrG7e2tsEwIUlaWTGmZiaGcslU9OPeX6Xxx9+3eH0DumeQgghhDj5Jk8u5p13NnDzzbcxd+48PB5P+B+AJUs+x5TbHiE2b/pJHunAjk4uHa3D6WPdwdZ+vzNolfBCY38gSOAY1QV6e2tPI9XtLspbnby5p5GkaAPLitPIiDOyYHwi5xYNnDAKBFXs3RMq3oBKY+cgDyCEEEKI08KauxZyzskexBAMNlNU3eHmQEv/C42j9D1Tmf7g0DfArS1tosHmprTBzuu7Gzl3QgoLxiWSEWdkWXEqidEDL47ucPnC8052t1/mnYQQohfZwXQSHEkyNdvd+PwqWd29jU6XlSb9mZAajcsX5KIpaeg0Cpsr2ilIjmH+uKFtG/YHesKLhO4GDKtmZrFq5tBWoqyemcXz2+qINmhZPStz+A8ghBBCiJPGYonll7/8DV6vl2CwJyZQFAWjsWd1aHW7k2iDlqTu3kSna+ykUyAnMQpFUbjunBwONHVR1uJgbn4CcVFD2L4EOHvVM06JMaAoCncsHTekc/VaDZfPyGDDwVYy4kyDLuQRQgghxOnjx91zTkFVpbzFQXqsiRhTaPrvdI2dLEYtCWYDZoOW284t4M09TbQ7vFw8bWgxjKqqeLsXQGsUyE0wo9dq+OHFk4Z0/oTUGC6Zms7eejtTMmMpSo057mcRQogzjSSYTqJUS2SZl/uX5/PQ2srw+zV3LeSyRzb2qel/KirOsLC1ysbfPqokL9HM/1w9bVg1cA+39HTvzks0D/v+C8YnsWC8lHYRQgghTmcGw7HL6uYeFSPEAp293q+5a+FpMXHiV+Hiqek8s7mGX7x5gHkFCfx05eQhN6a2uXzh8sEKMDHdMuwxDGchjxBCCCFOPxpFoXCQRMiLt89l9eOfjtGIjp/bE+DaJZn8fkMF972wm6tmZ/GN8wuHfP7minZau0K7jgw6DbPy4od1f0UJlSQWQgjRlySYTiHzJmSyZkLk7ptXu1eevPRZNU9sqj0ZwxqS3iXzKtqcbDjQwrJhNIzWaXomVMyG4Te2FkIIIcTZ55/9lB4+slP8VE80Pfp+Rfj1pvIOOt3+Ie9e0ioKOq0GCKDTKpj0UvVaCCGEEIPrr23DmrsW4vf7WXUK9wz3Ab9+tyz8/j9b67luXt6Qz9drNWg1CsGgilmvRaMZ2qIeIYQQg5P/Gj1NrJqde7KHMCwmvcLzn9WypaI9/JkvEGRXrRW7y9fn+FuXFLCkMInVMzNZOT2UZOt0+Xjo9X1874XdbCprCx/r9Ppp65J6t0IIIYQY2OnQ97I3m9PLf7bWcKDBHv7M6fWzq9aKxx/ZHzPGpOOWRfksKUziS/NzmZGTAEB5Sxf3v7yX+1/eS1lzV8+1XT5s/cRfQgghhBAAutOkZ/gRKioHGuz8Z2sNjTZ3+PN2h4e9dZ0Ej+ppOTsvgRvm57CkMImbF+cTYwytt/+kvI3vvbCbh17fh7VXX6W2Li9Orx8hhBCDkx1Mp5HeP/an8qpcvVZhbWkL22tsmPQavrmskAXjkvjZa/vYXmMjO97ETy8rJi3WRE27k921Nt7c20iXO0C82YC2eyXJ85/VsrmiAwBfoI4F45PYU2fjt+8ewuryc9n0DL68YOgrVoQQQghxdjldYqeseAO/XnuQqjYXr5kbeeiKYuKjDPzo5b1UtjkpzrTw0OUl6DQKh1u62FLezoeH2/AFghHl8f6ztY6dtbbQ689q+cGKSawtbeIfG6sA+OrCPC4Yxg5zIYQQQpxdjsROgWCQKx775CSPZmDz8hP4+Zv76XD6+PBQG/9z1TQqWh386q0DtHZ5WTohmXuXT8AfCLKv0c4Hh1rYUW1Dp1H43MSU8HWe21rHwe5FORlxddy8uICnNlXx6s4G4qJ0fGtZEVOy4k7WYwohxGlBEkynofX7Ggc/6CS6+7xx/HF9qOyL2xekvNnB5HQL+7pX5NZa3Wyt7MAbCPLUpir8Pb28ef9AMzfMz8Fs0BHbq0xMlCG02W5jWTtN9tCqks+qrJJgEkIIIcSgbn5y88kewjFdPDWDv34YSgJ1OH0cbnKgURxUtjkB2N9gp8Xu4ZUd9byxpyni3HX7m7l8Rmj3d7Sxp8ywWR96vaWinU53aAXulsoOSTAJIYQQYlCncnIJQiXvOpyh3dkNVjd2t59PevVZ2t9gJ6iqPPT6fj6rtkacu7a0iXMKEgEw94qdYk2hOaitVVY8/iDNdi+bytslwSSEEIOQBNNp5lRefQtg0sIf1pXjC4S2I6fE6FlWnMqfNlTg6c4kZcYamVeQwI9e3huRXAJIsRgx6kI/8KtmZuEPqrQ7vKzubkJdkhnLuv3NOLwB0mONY/dgQgghhDgtneqx0/z8WP76YRVHCrlMzohhVm48D76+L3zM5HQLCWY9Gw629Dk/xdITD926pACzQYuqwvXzQuWVJ6RZ+Kyqg2AQMuNMo/osQgghhDj9neqx03kTk1h3oKeNwuKiJDpdXj442Br+bHKGhfoOF9uOSi4BpPeKh+76/Hhe2FZHotnA6lmheadxydFUtDow6RQKkqJH8UmEEOLMIAmmM1TvkjCNXV3c+vddY3LftFgTVR099W8LkqIx6rRsq7aGJ04unpbB+n3N1Ns8fc6fm58QLpGn1ShcOzcn4vtFhUl0OLz8c0s1W6o6+PvGKr66UHYxCSGEEOL4XZoDt1/eEzt9sL+R/363fEzuvbuui95dAj4/IZVtNVYONTsA0CqhyY//evsgDm/kyhwFuKikZ0eSSa/l5sUFEcdcPSeb9i4P7+5rYe3eJrLio1heIruYhBBCCHH8ju7X9MMXd7C73jnq99UB7/dKLgHcdm4BL2yro9kemmNKijbwrQuKuP3pbahHnR+l13Dt3Ozw+7RYE1/73PiIY+4+bzw2l5ft1Vb+uaWa9Dij7GISQohj0JzsAYiRs+auheF/ekuPiRmTZo03zs+krldzRYALStKpaOkiNyEKgLRYI3PzE1h/qLW/S2DSa/v9vLdaq4suTwB/QGVnTd/VKEIIIYQQgzHSEzv1Ti4BnDspfUxip+9fWIjD15M00ihQnBmDVlFIjzUAMC4lhrQ4Ewe6Sw33plFAUZRB79PS5cUTCNLlDfBZVcfIPYAQQgghzhpfmJbS75wTwC9Wz+Af100a9TGsnp1F7+U20XoNHQ4vsSY9lu5yd0WpMWgUhdauvouaDToNqMeOnbQahZYuLwEVWru8fFLePpKPIIQQZxzZwXSGGMokyJFjbDYbf/ighs1VnSM6hld2t0SUvLuwOIVnP63hcIuDcclm7lhawKzceDLiolg4Polnt9ZFnF+cEcMV3T0EjmVqdhwbDrbg8AQoSosZ0WcQQgghxNnh+WHETtU2G0+NQuz0u3U9u6Q0wJfn5/Cz1w7QZPcwIyeOK2dns2B8EnqthoJkM7vrI5NMK6akMTsvYdD7TEq3sL3GiqLApAzLiD6DEEIIIc4OXz236JjfJyYmhmOnrZVtvFvaxMflI7so+MXtPfNIRq3CVxbk8t3nd+PwBlhSlMyMnDiWTkgBIC5KT5vDFz5eAW44J5cY0+BToUWpMVS1ObGYdEzLlt1LQghxLJJgOs0c+bE+kZq4cXFx3L8y7oSvc7T2Xj/cy4tTWTAuibdLQ/0Dylud5CdFkxEXRafbx+u7G/ucn5cYjU47+Ka6ReOTyIo30dblZVZu/IiNXwghhBBnnv5ip+HuTsrtjp1qamr42is1IzKu7Dg9tbae2OneC4uobnfR1F3epbLVwQOXTkav1XCw0c7+pq4+1zgygTKYq+ZkU5JpQVE0TJYEkxBCCCGOYSRipzn5SczJT+IPa/ez9uDI7ADKijNS191qQauBP14/g2e21GL3BABosLq5d/kEAF7dUR+RXAJQgWXFqUO6193njWfh+ETSYk3kJppHZPxCCHGmUlT16IqkPVpa7AN/Kc44J5Js+t6FE/ivtw8CoNMozM6NZXOlLfz9xLQYHrqiBJNey8PvHGLdgb5Nqv9961yijfrjHoMQQpzNUlIsg9fJEqNOYqezy4nETlfNyuS5bfUARBu0JJp11Fh7SrksHJfIDy4OlZr5zn92crC7J1NvY1HGTwghzkQSN506JHY6uxxv7JRo0jAuLZatVaEdUemxRjpdPpzdpYYVYNXMTG5clA/AF//yCV2eyN6VGuAViZ2EEOK4HCt2kh1MIuzoSYrPyjv46Rv7hnxem8PD7tpOVkxJ46ev7Y845uerSjDqtPgDQTZXtPW5RlyUrt/kUoPVzd83VgLw1YV5ZMRHDfVxhBBCCCFG1dGx0z1Pf0K5LTjA0ZHnBYIqQRXqrW6un5/DXf/aGf5eo8B9F00EoKbD2W9yaU5u/+VadtfZeHFbHRaTjjuXjifKMHh/SyGEEEKIsXB07DSUhJNFC/+4ZT4Oj48nPq7C7Q0wOTOGxz+oCh+TYNaFk0vvH2zpk1wC+PrScf1e/719zXxwqIWcBDM3Lc5HM4T+lkIIIXoMXo9MnLVmj0vgqesnH/OYc/NM4deXz8ji/ksnMzs/sc9xRl1ocsMXUHF4+/7QP33T3H6v/8zmajaWt7OxvJ2nP6kezvCFEEIIIcbU7740n8Gq9L94eyjm0WoUvroonx9eMgmzIXLNl6b7e4AGq7Pf6/zkspJ+P39qUxVbq6ysP9DKs1trhzV+IYQQQoixNJTd2P93Z+iYaKOeu88r5LsXTaSmzR1xTLSxJ5Y63Nh/z8zlU9P7fOb1B3lmczXbqm28srOBdfuahzN8IYQQSIJJDCIh4diNo7+7cla/n49L7tlpdMvC7PDr6/68ud/jlX5WiDTa3Oys7WkIeWSiRQghhBDiVPXMURMlxl6vk/Sg1/fdsZ0YbSDG2BOWP3r9zPDrn71+sM/xSeb+Q/idNVaq2noSUnqJnYQQQghxirtlYW7E+96R0pdmJvV7znXzciMmNP/wxZ7Y6ZVdTX2O/+r8rH6v897+ZqxOLxDaQW7QyTSpEEIMl/RgEkPS2NhIenrf1R4DUVWVBpubuCh9eCVJVaudu/69u8+xz952Tp+Vuy12N/e/vJf67gaOsSYdj3xxBgnRhhN4CiGEOHNJL4FTg8RO4ojhxk6BoEqDzUWKxRje+f3j/2xiR3Pfv1L9rfYtrbPxX2sP0t7d0DozzsQj181Ar5WJEiGEOJrETacOiZ3EEcONnTz+AC12DxlxUeEFyQOV3OsvdtpwoIU/bigLV9kpzrALOfJDAAAZt0lEQVTwq9VT+l0ALYQQZzvpwSRO2HB+5CG0IynzqH5J/SWXzDr6JJd8gSA3/WNbxGeTMyySXBJCCCHEaWO4sZNWo5CdYI74rL/k0oxMS5/P9td38L2XIvtmzs6Ll+SSEEIIIU4bw42djDptn9ipPz+4sKjPZ39+v4w1e3p2OinA0gkpklwSQojjIAkmcVI9e0ffVSSr//hJxPsYo4ZvnF84VkMSQgghhDjpNpe39vlsZXESt503sc/n330xMrk0PtnMTd2NroUQQgghzgb97V566saZJERHLn72+nwRySWA1bMyWDElbVTHJ4QQZypJMImTZijNHAEmplmINfXtV3AitlZ28NSmKox6DbctKaAore9qYCGEEEKIk2VrRVufz/pLLjm9/v/f3n2HR1WmfRz/pRBImxQSTCMESCCwkhCQogjqou7SFCkr4KqwoiIlIiCCoCsIiiBdFEEJRQIiCAgviK7u0kvoXTqBJJBACKmkzvsHu4NDIjE4YYbw/fyV5z5n5twz18XFPec+z/MUi7UIrSpHC89eWr0/SWsPJMnbrbKGPBkmD2dmlgMAANt2c3NJkuZuiS8We6i2j8VnL3256Yx2nk1VLR9XDXqiDvuKA6iwWDcDVvFb/63O2nCiWKy6d/GC4I+a+vNxnb6craMXMrVsd4LF3x8AAOCP6NfavJn0yoMlLxszbs3hYrHQapZ9cKagsEhzNp9R/JVr2nvuqr7bm2TR9wcAAPijpv+9vtl4wU1jSSooKtK2k8VniQd6VrFoLieSM7Rib6LOX7mmDccva+Px4tcEgIqCGUy4Y0qbsfTd3gSt2p9cLJ6bX2ixHK7lF2r0qiNKy77xtG/GteJP/gIAAFhbabXT8GX7dTAps1g841qexXK4mH5N76w4pPzCG/tB8QQuAACwNSGenqXWTs9/uUOZuUXF4nkFRrlWtkwehxPTNXq1+QNAAR6WbWABgC1hBhNsxpKdJc8k6tY02GLXmPKvYzqQmG4a20nq1iTIYu8PAABwpxy9WLy5ZCfp4TBfi13jwzVHlZSeaxo72tvpqcgAi70/AADAnVJSc6myo528XC2z9G9+YZHGfX9UWXk3rmOo4qg6fmzLAKDiYgYTbEaQl7OuJmWYxZrV8JS3hR4jyc0v1NaTV8xiUcGeahDkKUn6dneCdpxOVS1fV73csqbF198FAACwJOdKDsrINZ/p/VKLGrK3UA1z8mKGTl7KNot1ivKXWxVHGY1Gzd54WqdSstS0prc6NQq0yDUBAADKi4Od9KtJ2bKTNKVrA4u9/5K4c7qSbb5KzqAnQiVdX1Fn+s8nlZqVp3YN/PRwmI/FrgsA1sQMJtiMYW3qyvG/90PsJY3qEK6RHYqvmXu7us/erpufVfnfNOWktBwt2B6vQ0kZWrX/gjafKL6xNgAAgC3p92hN098ele218KUmejrKco2egd8cKBZzq1JJkrTmwAWt2n9Bh5IytGTneaVlW25ZPgAAgPLQqdGNWdi1fV20vG9zBVV1s8h7Z+Xma3EJK/O4/7d2mrnhlDYcv6SDielaHHfeItcEAFvADCbYDE8XJ0W3DtXOs1cU7ueuRjW8Lfbe/z6SrPybukuuTvZqWN1DkhS7I14F/32Mxd5O8nCuZLFrAwAAlIcWYdX0YnqeTqVk6eFQHxksWL+8uXR/sZifu5OahHhJktYduWiKGyU5OfLcGgAAsG09mgYrv9Co1Ox8PR3pLwd7y9UvvWJ2FouFVXNTsLeL8guLFHf6xoo6hUXFl+oDgLsVDSbYlMfCq+mx8GoWf9+DSWnFYpP+FqkAT2ddzszTiZQbexiEVHVRgyAPi+cAAABgaV0al89ekolpOWZjjyrS+K4R8nJx0tnLmbqaeWNfpohAD7k48bMCAADYNkcHe730cM3ST7wNOQVGs3EDfze90+FPqlLJQTtOXVJO7o2l8x6qXbVccgAAa+CXIO4JW09cMhs7O0gBns66cDVbLy/Ya4rX83NTn0dq3en0AAAAbEr6NfO9nRoF+8jLxUmr9iVo1sazpnizml4a8Ofadzo9AAAAm9a6vp+cnRw06rtD2hl/VdL1PZ/+HO6rHs2CrZscAFgQDSbcE+ztHSTduFHSOer6urvTfz5pdl4tHxfV8rXM+rsAAAB3K3vJbO/KV1qFSJJit58zOy/6z7UtujQfAABARdC63vXVefacv2qKGSUNfDzMShkBQPlgsXTcEyZ0iTD9XcleerZ5iCSpSYj5Pk/tIgIEAABwr3vt0RvLx9zn7ii3Kk6SpNq+7mbnGZyd7mheAAAAtuiRsBv3l1rW9jT9/es9vh3t7e5oTgBwJ9gZjcbfPJiSkvHbB4EKwGg06vP1JxV3Nk3PNPRX+8hAa6cEALfF19edXys2gNoJFV1+YZHGrT2qs5ezFd06VBFBnqW/CABsDHWT7aB2QkV3OTNX477/RRnXCjT26Xqq6u5s7ZQAoMxuVTvRYAIAoALgRoltoHYCAMD2UTfZDmonAABs361qJ5bIAwAAAAAAAAAAQJnQYALKWXL6NQ3/9oBiNp/RrWYMAgAAQNobf0VvLTugNfuTrJ0KAACAzVu9L1HDvj2g3fGp1k4FwD3I0doJABVZdm6BXpq/W5J0MDFD2XkF6vdYqJWzAgAAsE2HE9L0zndHrv+dlCGXKg56tE41K2cFAABgm+ZsPKnl+y5Kkt777qjm/eMBebk4WTkrAPcSGkxAOeqzcLfZ+HBShpUysbyCwiJ9ufmMUjPz1D7CTw3Y5BsAAPxBb684bDbeeTqtwjSYrmbna87mMyooMurvzarL35NNvgEAwO0rKioyNZckySjp7KVseQVXjAbT6UtZWrLzvFycHPRSixC5VOY2NmCL+JcJlKMr2QVm4xcerG6lTCxvxd5Erd5/QZJ0MSNXU56lwQQAAG5fTl6BCm9aTfjlliFWyaU8zN92Vj//kiJJyiss0oi24VbOCAAA3M3+dTS5WCwiyGCFTMpHzJaz2hOfJkmqUslBL7esaeWMAJSEPZiAcuRc6cY/MUMVBzWr6WPFbCzr19tJsbUUAAD4oyo52Mve7sY43M9VHhVoiRfjbw4AAADKLtjbxWzcq3mw7O0rzq3eX+9jzp7mgO1iBhNQjj7uGqHJPx6Xh4uj3mlXz9rpWNQzUQG6lJmry1l5ah/hb+10AADAXc7RwV7D/lpHi3acU5ivmwY8HmbtlCzq+ebByi8oUkGRUT2aVpxZ7QAAwDrC/Qx6oXmwNh5PUYvQqur0QJC1U7Kong/W0BKn86pSyUHPNQu2djoAfoPdrTrAKSkZtIcBALgL+Pq625V+FsobtRMAALaPusl2UDsBAGD7blU7VZx5kwAAAAAAAAAAALgjaDABAAAAAAAAAACgTGgwAQAAAAAAAAAAoExoMAEAAAAAAAAAAKBMaDABAAAAAAAAAACgTGgwAQAAAAAAAAAAoExoMAEAAAAAAAAAAKBMaDABAAAAAAAAAACgTGgwAQAAAAAAAAAAoExoMAEAAAAAAAAAAKBMaDABAAAAAAAAAACgTGgwAQAAAAAAAAAAoEwcrZ0AANtxOPGqZm44raquTnqnXbjs7elBAwAA/Ja1B5P0f/svKNzfXf0fC7V2OgAAADZtzubT2n02TY/W8VWXB4KsnQ4AC+DuMQCTUauO6PSlbO08m6YP1/5i7XQAAABsVnZegT5ff1pnU3O07lCylu06b+2UAAAAbNaWk5e0fE+SzqbmaP62eCWl5Vg7JQAWQIMJgCRp5IqDys4vMo1Ts/KsmA0AAIDtKioqUnTsXhUab8TiU7OtlxAAAIANy8ot0MQfjpvGRkkpmdeslxAAi2GJPACasO6I9p1PN43t7aSBT7DMCyqegwcPqF+/3po8eYYaNXpAkvTzz//S3LmzlZiYID8/f3Xv/rzatXvK9Jr8/HzNnPmJfvzxe127dk2RkQ31xhtDFRAQWOI1CgoKNH/+HH3//f8pNfWygoNrqFevl9Wy5aMlnj9x4kfaunWTli5dZfHPCwAoHy/GxCktp9A0dna0Vz+WyMM9JDn5oqZNm6Rdu+JkNBapWbMHNWDAIPn4+Eoqvb662enTpzR9+mQdPLhfTk6V9Mgjf9Zrr0XLzc1NUtnrKwCAbek+e4d+9VyOang7KyLIy2r5AHez/Px8zZkzSz/++L3S09MVFdVIAwYMUlBQddM5q1ev1KJFC5SUlKiAgMBSa7E/ghlMALTh+BWzcYcIP1X3crVSNkD5yMnJ0Zgx76qw8MYNwX379mj06JHq1Olvmjdvsbp06abx48dqy5ZNpnMmTPhA//73v/TPf47RzJlfKjc3V8OGDZLRaCzpMpo161OtXLlM0dGDNXfuIj322OMaMWKo9u7dXezc7du3avnybyz/YQEA5eZ0SqZZc0mS3vxrHTk58tMK9waj0ag33xyojIwMTZs2U9Onz9Lly5c1dOgbkn5fffVr2dnZGjiwrwwGg2bPnqdx4yZp3769+uCDUaZzylJfAQBsy6jvDurmX88fdW5glVyAimDSpPFasWKZXnstWrNnz5OPj6/69u2ttLQ0SdJ//vOTJk4cp+eee1FfffWNnn32OY0fP1abNq0vl3z4FQTcw4qKivTlpjPF4r1b1rrzyQDlbPr0SfL1rWYW27hxvWrVClXHjp0VGBikjh07q06dutqxY6skKSHhvNasWaURI95T48ZNVKtWqIYMGaasrCwlJBTfa8NoNGrVqhXq2fNlPfxwKwUFVdfzz/dSVFRjrVljPkMpPf2qxo17Xw0bNiq/Dw0AsKhreQX6fP2pYvEmId5WyAawjtTUywoJCdGwYSMVFlZHYWF19OyzPXTs2FGlp6eXWl/d7MKFJEVENNRbb41UjRohuv/+CD311DPatWuHpLLVVwAA25KcnqO9v1oxR5Jq+1SRa2UW1QJuR3p6ulavXqG+faPVuvUTqlEjRIMHD5Orq6u+/XaJJCkt7Yr+8Y9X1LZtBwUEBKpDh46qVau2du6MK5ec+NcM3MOe/nRbsdgrLaqXcCZwd9u6dZO2bt2sjz+ephdf7GaKe3p66syZU9q9e6eiohpr3749OnXqpDp1+pskKS5umzw9vdS4cRPTa4KDQ7Rs2eoSr1NYWKjRoz9U7drmyyTZ2dkpIyPDLDZhwodq0aKVfHx8tHr1SrNja9as0sKF85SYmCBv76pq27aDevV6Wfb2PBcCANaSm5urrrN3FYvP6/mAFbIBrKdqVR+NGvWhaZycfFErV36revXqy2AwlFpf3axWrdp6//1xpnF8/FmtW7dGTZo0l/T76qucnBxNnjxeW7duVlZWpkJD6+jVV/uZ1XAAgDurwydbSoxP6cZDlsDtSkg4J6PRqMjIhqaYvb29QkPrmGZ2d+zYxXSsoKBAGzb8R2fPnlHv3q+Z4l99NVcrVy7XpUvJuu8+f3Xt2k2dO5dcq5WGBhNwjyrpP/pV/R+yQiZA+UpLS9O4cWM0fPi7cnd3NzvWqdPfdODAPkVH95GDg4MKCwvVvfvzatOmvSTp3Ll4BQQE6ocfvtfChfOUlnZFDRpEKjp6kKpVu6/YtRwdHdWkSTOz2JEjh7R7904NGvSWKbZu3RodO3ZUc+cu0tdfLzQ7/8SJ45ow4QO9995Y1a1bX7/8ckSjR49UQECgKS8AwJ3XpYTmErUT7nXDhw/Wxo3r5e5u0PTpn0sqvb66lZ49e+jEiWPy8/PXhx9+LOn31VdffDFTp0+f0qRJ0+Xi4qpFixZo+PAhWrnyezk7O1v4UwMAbsdDtbw1vG24tdMA7mr/2+/y4sWLCg4OMcUvXEjUtWvXzM49evSwXn21lwoLC9WhQ0c99NDDkqRNmzYoNnaBRo/+UEFB1RUXt13jx49V7dqht7XKDo9CAwAqtAkTxqpFi5Zq3rz4TcArV1KVmpqqvn2j9cUX8zVw4BB9++0S04yirKwsxcef0eLFXyk6epDef3+crlxJ1euvv6bc3NxSr33+/Dm9/fabqlfvT2rf/mlJ0sWLFzR16kS9/fZ7Jd7wSEg4Lzs7O913n7/8/Pz0yCOPacqUTxUVxRPyAADAtrz0Uh/NmjVXERGRGjiwr1JSkkutr25l+PB3NWPGbPn4+Co6uk+xGyVSyfVVQsI5ubi4yt8/UIGBQerXb6DGjh3P7G8AsCHPNWPFHOCP8vWtpsaNm2jGjKk6dy5eBQUF+uabxTp27Bfl5xeYnevvH6Avvpiv4cPf1U8//ahZsz6VdL1uqlTJUX5+/vLz81eHDh01ZcqnqlEj5LZyotoC7lE3P3HLE7ioiNauXa1jx46pf/83Sjw+fvxY1alTVz16vKCwsLrq0qWbund/Xp99Nk1Go1GOjo7KzMzUmDEfqXHjJoqIaKgxYz5SQsJ5bdu2+ZbXPnr0iPr27S2DwaDx4yfL0dFRRqNRH3wwSu3aPWU2nfnXmjd/UPXr36/evZ9Xt27PaNKkj1RUVCQ/P78//H0AAG4ftRNQXGhomOrXv1+jRn2ooqIirV27utT66lbq1g1XZGSUxowZr8TEBG3Y8B+z4yXVV5LUvfsLOnbsqNq3f1z9+7+ipUu/Vs2atVS5cuXy+ugAgFLY3TQOrupqlTyAiuadd0bL09NTPXp0VuvWLbRr1w61bdtBbm5uZud5eHgqLKyu2rV7Si+80EtLlsSqsLBQTz7ZRgaDh7p1e0YvvthNM2ZMlcFgkJfX7e0ryxJ5wD2MGyOo6NasWaWUlIt6+um/SJLppsaQIa+rTZt2OnTogB5//C9mr6lf/37NnfuFMjIy5OPjK2dnZwUEBJqOe3l5y2DwUGJi4m9ed8eObRoxYqhCQ8P00UeTZTAYJF2fvbRrV5wOHtyvFSuWSrq+Hm5BQYGeeKKlPv54miIjo/TJJ7N09Ohhbdu2Rdu3b9Hy5UvVp09/Pffcixb9fgAAZUPtBEipqZe1e/dOsxqqSpUqCgwMVEpKSqn11f/qov9JSkrUiRPH1LLlo6aYj4+PDAYPXbqUbIr9Vn0lSZGRDbV8+Rpt375VcXHbtHz5N/r66680ffos1axZy8LfAADg9/iOugkoFz4+vpoy5VNlZmaqqKhIBoNBb7/9pgIDr9+72rNnl9zc3BQWVtf0mtq1Q5Wbm6v09HR5eXlr3rzF2r9/r7Zv36pt2zZryZJYjRgxSk8++dcy58MMJgBAhfXuu+/rq6++UUxMrGJiYjVx4ieSpGHDRqp37z7y9a2mkyePm73m1KkT8vDwkMFgUGRklHJycnTmzGnT8cuXL+nq1TQFBgaVeM19+/Zo2LBBiopqrClTZpjd/PDx8dXixcs1b95iU06dOnWVj4+vYmJiFR5eT3Fx2zR37hcKD6+vnj1767PP5ujppztr7drV5fANAQAAlM2FC0l6770ROnr0sCmWmZmp+PizCgmpWWp9dbPDhw9p5Mi3lJp62RRLTExQWtoVhYRcbw7dqr6SpJiY2TpwYK8eeeQxDRkyXIsWfavCwiJt2bLRkh8dAADAqoxGo4YMiVZc3Da5ubnJYDAoKytTu3fHqUmT5pKkhQvnafbsz8xed/jwIXl5ecvT01M//fSDli9fqoYNG+nVV/spJiZWTZo00w8/rLmtnJjBBACosHx9q5mNnZycJF1v9Hh5eatr1+6aPn2SQkJqqmnTB3Xw4AEtWBCjnj1fliQ1bNhIkZFRGjVqhAYPHqYqVZw1bdpEBQfX0IMPtpAkZWdnKycnW1Wr+igvL0+jRo1U9erBGjz4LWVmZiozM1OSVKmSkwwGg4KCzNeddnc3yMHBwRR3dKykmJjZcnV1U4sWLZWaell79uzUn/7UoFy/KwAAgN8jPLy+IiOjNG7cGA0d+rYcHR01c+Yn8vT0Ups27WVnZ3fL+kqSrly5okqVKsnN7Xq94+8fqFGj3lF09CBlZ2dp8uQJuv/+CDVv/tDvqq+SkhK1bt0aDR06QgEBgYqL266srEzVr3+/tb4mAAAAi8nMzFR+fr68vLxkMHhoxoxpGj7cU46Ojpo8ebx8favpL39pK0l69tkeGjRogGJj56tVq8e0d+8uxcbO14ABb8jOzk55eXmaMWOq3N3dFRHRUOfPn9OxY7+oY8fOt5Wb3a3WQE5Jybj1AskAANxFkpMvqlOndpo2baYaNXpAkrRq1QotWRKrpKRE+fkF6JlnuqhTp66ys7u+YnRGRoZmzJii9ev/rYKCAj3wQFO98cabqlbtPknSl19+rpiY2dq0aad27NimQYP6l3jtxo2baurUT4vF5879QqtXr9TSpatMsbVrVys2dr4SEhLk6uqqVq0eVb9+r8vF5bfXrPb1db95iWtYAbUTAOBekJaWphkzpmjr1s3Ky8tT06bN9frrg00P95RWX3Xp0kFRUY01YsR7kq4vkzdt2iTt2bNLdnZ2atXqUQ0YMEhubm6/q77Kzs7WJ59M1ubNG5WeflWBgdX13HMvqE2b9iW+jrrJdlA7AQBQurFj39OePbu0dOkqZWRkaOrUj7VlyyZJUvPmD6l//4Hy9q5qOn/9+p81Z85snTsXr2rV7tPf//6C2rfvaDoeG7tAK1cuU3LyRXl5eevJJ9uod+8+pv0tb3ar2okGEwAAFQA3SmwDtRMAALaPusl2UDsBAGD7blU7sQcTAAAAAAAAAAAAyoQGEwAAAAAAAAAAAMrklkvkAQAAAAAAAAAAADdjBhMAAAAAAAAAAADKhAYTAAAAAAAAAAAAyoQGEwAAAAAAAAAAAMqEBhMAAAAAAAAAAADKhAYTAAAAAAAAAAAAyoQGEwAAAAAAAAAAAMrk/wG5qv69ijywDwAAAABJRU5ErkJggg==\n",
   "text/plain": "<Figure size 2160x360 with 3 Axes>"
  },
  "metadata": {},
  "output_type": "display_data"
 },
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "time: 11min 33s\n"
 }
]
```

___

# Evaluasi

Bandingkan dengan dataframe **fraud** yang berupa data fraud berdasarkan rule
base untuk masing-masing hasil prediksi. Cek mana yang paling cocok atau paling
banyak menangkap nomor telepon fraud yang sama.

```{.python .input  n=111}
fraud.set_index(['source_num'], inplace=True)
```

```{.json .output n=111}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "time: 4.96 ms\n"
 }
]
```

```{.python .input  n=131}
data_predict_svm_1 = load('dataset/data_predict_svm_1.joblib')
data_predict_cov_1 = load('dataset/data_predict_cov_1.joblib')
data_predict_isofor_1 = load('dataset/data_predict_isofor_1.joblib')

data_predict_svm_2 = load('dataset/data_predict_svm_2.joblib')
data_predict_cov_2 = load('dataset/data_predict_cov_2.joblib')
data_predict_isofor_2 = load('dataset/data_predict_isofor_2.joblib')

data_predict_svm_3 = load('dataset/data_predict_svm_3.joblib')
data_predict_cov_3 = load('dataset/data_predict_cov_3.joblib')
data_predict_isofor_3 = load('dataset/data_predict_isofor_3.joblib')
```

```{.json .output n=131}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "time: 3.41 s\n"
 }
]
```

```{.python .input  n=112}
# Cek jumlah nomor telepon unik untuk masing-masing dataframe.
print(fraud.shape)
print(data_predict_svm_1.shape)
print(data_predict_cov_1.shape)
print(data_predict_isofor_1.shape)
```

```{.json .output n=112}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "(149, 11)\n(4225, 9)\n(4423, 9)\n(5390, 9)\ntime: 973 \u00b5s\n"
 }
]
```

```{.python .input  n=120}
# Cek jumlah nomor telepon unik untuk masing-masing dataframe.
print(fraud.index.nunique())
print(data_predict_svm_1.index.get_level_values(0).nunique())
print(data_predict_cov_1.index.get_level_values(0).nunique())
print(data_predict_isofor_1.index.get_level_values(0).nunique())
```

```{.json .output n=120}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "69\n4225\n4423\n5390\ntime: 27.7 ms\n"
 }
]
```

```{.python .input  n=168}
# Cek jumlah nomor telepon unik untuk masing-masing dataframe.
print(fraud.index.nunique())
print(data_predict_isofor_1.index.get_level_values(0).nunique())
print(data_predict_isofor_2.index.get_level_values(0).nunique())
print(data_predict_isofor_3.index.get_level_values(0).nunique())
```

```{.json .output n=168}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "69\n5390\n3977\n5636\ntime: 3.96 ms\n"
 }
]
```

Merge (union) data dari masing-masing hasil model prediksi untuk menyaring data
anomaly lebih banyak.

```{.python .input  n=122}
data_merge_svm = data_predict_svm_1.merge(data_predict_svm_2, how='outer', on=['source_num'])
data_merge_svm = data_merge_svm.merge(data_predict_svm_3, how='outer', on=['source_num'])
print(data_merge_svm.index.nunique())

data_merge_cov = data_predict_cov_1.merge(data_predict_cov_2, how='outer', on=['source_num'])
data_merge_cov = data_merge_cov.merge(data_predict_cov_3, how='outer', on=['source_num'])
print(data_merge_cov.index.nunique())

data_merge_isofor = data_predict_isofor_1.merge(data_predict_isofor_2, how='outer', on=['source_num'])
data_merge_isofor = data_merge_isofor.merge(data_predict_isofor_3, how='outer', on=['source_num'])
print(data_merge_isofor.index.nunique())
```

```{.json .output n=122}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "15176\n15239\n9613\ntime: 492 ms\n"
 }
]
```

Lanjutkan dengan melakukan komparasi (irisan) data nomor telepon asal yang unik
terhadap data fraud berdasarkan rule base.

```{.python .input  n=123}
compare_svm = data_merge_svm.merge(fraud, how='inner', on=['source_num'])
compare_svm.index.nunique()
```

```{.json .output n=123}
[
 {
  "data": {
   "text/plain": "42"
  },
  "execution_count": 123,
  "metadata": {},
  "output_type": "execute_result"
 },
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "time: 72.9 ms\n"
 }
]
```

```{.python .input  n=124}
compare_cov = data_merge_cov.merge(fraud, how='inner', on=['source_num'])
compare_cov.index.nunique()
```

```{.json .output n=124}
[
 {
  "data": {
   "text/plain": "42"
  },
  "execution_count": 124,
  "metadata": {},
  "output_type": "execute_result"
 },
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "time: 19.8 ms\n"
 }
]
```

```{.python .input  n=125}
compare_isofor = data_merge_isofor.merge(fraud, how='inner', on=['source_num'])
compare_isofor.index.nunique()
```

```{.json .output n=125}
[
 {
  "data": {
   "text/plain": "40"
  },
  "execution_count": 125,
  "metadata": {},
  "output_type": "execute_result"
 },
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "time: 39.9 ms\n"
 }
]
```

___

# Kesimpulan

Dari hasil model **Anomaly Detection** di atas dapat dilihat **Isolation
Forest** lebih sedikit menangkap anomali data. Dan jumlahnya tidak terpaut jauh
pada saat dilakukan union data dari ke-3 data ready dibandingkan dengan
menggunakan model lain yang terpaut jauh. Akan tetapi ketika menangkap kecocokan
data dengan data fraud berdasarkan rule base lebih sedikit dibanding model
lainnya.

Untuk itu perlu dilakukan pengecekan lanjutan terhadap data prediksi ini untuk
mengetahui data fraud yang sesungguhnya, karna data prediksi ini bukan
menandakan fraud yang sesungguhnya.
