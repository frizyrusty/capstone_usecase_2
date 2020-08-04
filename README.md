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

```python
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

```python
colname = ['start','end','source_num', 'dest_num', 'access_code', 'org_dest_num', 'duration', 'dest_country', 'dest_country_status']
data = pd.read_csv('dataset/fraud_pstn_202003.csv', sep='\t', names=colname, parse_dates=['start', 'end'])
fraud = pd.read_excel('dataset/fraud_pst_maret2020_label1.xlsx')
```

```python
data.head()
```

```python
fraud.head()
```

# Exploratory Data Analysis (EDA)

```python
print(data.shape) 
print(fraud.shape)
```

## Understand Missing Values

Hal yang pertama adalah membersihkan data dari data-data yang tidak perlu
ataupun NaN data (missing value).

```python
# Copy data ke dataframe ke nama dataframe baru
data_clean = data.copy()
data_clean.isna().sum()
```

### Inspect missing `dest_num`
Melakukan pemeriksaan pada missing value pada kolom `dest_num`, untuk
digolongkan sebagai inputation error. Cari data lain dengan `source_num` yang
sama dengan oberservasi missing value pada kolom `dest_num` tersebut.

```python
cond1 = data_clean['dest_num'].isna()
cond1
source_num_missing_access_code = data_clean[cond1]['source_num'].unique()
data_clean[data_clean['source_num'].isin(source_num_missing_access_code)]
```

Dari hasil yang didapat, dapat diasumsikan bahwa missing value pada kolom
`dest_num` adalah kesalahan data dan dapat dihapus. Tapi terlebih dahulu di cek
keterkaitannya dengan kolom lainnya.

___

### Inspect missing `access_code`
Melakukan pemeriksaan terhadap kolom `dest_num` yang memiliki msising value
terhadap `access_code`, apakah juga berupa inputation error atau memang tidak
memiliki nilai.

```python
cond2 = data_clean['access_code'].isna()
dest_num_missing_country = data_clean[cond2]['dest_num'].unique()
(data_clean[data_clean['dest_num'].isin(dest_num_missing_country)].index == data_clean[cond2].index).mean()
```

Dari hasil yang diberikan di atas, nilai `access_code` yang missing terjadi pada
beberapa nomor `dest_num`, sehingga asumsi bahwa nilai tersebut dikategorikan
sebagai inputation error dapat kita abaikan, dan dapat diekslkusifkan
(diremove).

```python
# Remove missing access_code (berdasarkan hasil di atas)
data_clean.dropna(subset=['access_code'], inplace=True)
```

___

### Inspect missing `dest_country`
Periksa apakah missing `dest_country` berimplikasi pada missing
`deset_country_status`. Hasil dibawah menyebutkan bahwa semua missing
`dest_country`, juga missing `dest_country_status`. Lalu putuskan apakah akan
menghapus data dengan missing value tersebut.

```python
cond3 = data_clean['dest_country'].isna()
cond4 = data_clean['dest_country_status'].isna()
(cond3 == cond4).mean()
```

```python
source_num_missing_country = data_clean[cond3 & cond4]['source_num'].unique()
dest_num_missing_country = data_clean[cond3 & cond4]['dest_num'].unique()
```

```python
len(dest_num_missing_country), len(source_num_missing_country), len(data_clean[cond3 & cond4])
```

```python
data_clean[data_clean['dest_num'].isin(dest_num_missing_country)].head()
```

Dari hasil yang diberikan di atas, nilai `dest_country` yang missing terjadi
pada beberapa nomor `dest_num`, sehingga asumsi bahwa nilai tersebut
dikategorikan sebagai inputation error dapat kita abaikan, dan dapat
dieksklusifkan (diremove).

```python
# delete all missing value `dest_country`
data_clean.dropna(subset=['dest_country'], inplace=True)
```

```python
print(data_clean.isna().sum())
```

Simpan data clean ke dalam file untuk mempermudah proses di selanjutnya.

```python
data_clean.to_csv('dataset/data_clean.csv', index=False)
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

```python
data_clean = pd.read_csv('dataset/data_clean.csv', parse_dates=['start', 'end'])
```

```python
data_clean['day'] = data_clean['start'].dt.date
data_clean['day'] = pd.to_datetime(data_clean.day)
data_clean['week'] = data_clean['start'].dt.week
data_clean.head()
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

```python
# Fungsi untuk mendapatkan tetangga nomor pemanggil
def find_neighbor(source_num, neighbor, threshold=100):
    return neighbor[abs(neighbor - source_num) <= threshold].unique()
```

```python
# Fungsi untuk mendapatkan tetangga nomor pemanggil berdasarkan harian data
def find_neighbor_day(source_num, day, neighbor_num, neighbor_day, threshold = 100) :
    cond1 = abs(neighbor_num - source_num) <= threshold
    cond2 = neighbor_day == day
    return neighbor_num[cond1 & cond2].unique()
```

```python
# Fungsi untuk mendapatkan tetangga nomor pemanggil berdasarkan harian data dan nomor tujuan yang sama dipanggil oleh source_num
def find_neighbor_day_dest(source_num, day, dest_num, neighbor_num, neighbor_day, neighbor_dest_num, threshold = 100) :
    cond1 = abs(neighbor_num - source_num) <= threshold
    cond2 = neighbor_day == day
    cond3 = neighbor_dest_num == dest_num
    return neighbor_num[cond1 & cond2].unique()
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

```python
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

```python
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

```python
data_ready_1['source_num_neighbor_unique'] = neighbor_sources
data_ready_1['dest_num_neighbor_unique'] = neighbor_destinations
data_ready_1['avg_interval'] = intervals
data_ready_1['duration_neighbor'] = neighbor_durations
data_ready_1['avg_interval'] = data_ready_1['avg_interval'].fillna(0)
# data_ready_1.head()
```

```python
data_ready_1.to_csv('dataset/fraud_data_ready.csv', index=True)
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

```python
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

```python
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

```python
data_ready_2['source_num_neighbor_unique'] = neighbor_sources
data_ready_2['dest_num_neighbor_unique'] = neighbor_destinations
data_ready_2['avg_interval'] = intervals
data_ready_2['duration_neighbor'] = neighbor_durations
data_ready_2['avg_interval'] = data_ready_2['avg_interval'].fillna(0)
# data_ready_1.head()
```

```python
data_ready_2.to_csv('dataset/fraud_data_ready_day.csv', index=True)
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

```python
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

```python
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

```python
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

```python
data_ready_3.to_csv('dataset/fraud_data_ready_day_dest_num.csv', index=True)
```

___

# Modeling
Selanjutnya mulai melakukan tahap modeling dengan menggunakan **Anomaly
Detection** dari `sklearn`, langkah-langkahnya sebagai berikut :

## Load Data Ready

Load kembali file data ready yang ingin dimodelkan, kemudian buang data yang
memiliki `total_duration` nya 0.

```python
# Jika menggunakan Data Ready 1
data_ready_1 = pd.read_csv('dataset/fraud_data_ready.csv', index_col=[0], skipinitialspace=True)

# Jika menggunakan Data Ready 2
data_ready_2 = pd.read_csv('dataset/fraud_data_ready_day.csv', index_col=[0,1], skipinitialspace=True)

# Jika menggunakan Data Ready 3
data_ready_3 = pd.read_csv('dataset/fraud_data_ready_day_dest_num_2.csv', index_col=[0,1,2], skipinitialspace=True)
```

## Data Scaling

```python
scaler = StandardScaler()

# Jika menggunakan Data Ready 1
data_scale_1 = scaler.fit_transform(data_ready_1)

# Jika menggunakan Data Ready 2
data_scale_2 = scaler.fit_transform(data_ready_2)

# Jika menggunakan Data Ready 3
data_scale_3 = scaler.fit_transform(data_ready_3)
```

```python
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

## Dimensionality Reduction

Menggunakan elbow method seperti di bawah ini untuk menentukan number of
component yang digunakan nantinya pada saat scaling.

```python
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

Dari hasil plot elbow method di atas telah didapatkan number of components yang
bisa digunakan, dalam kasus ini digunakan pc=8. Tapi untuk visualisasi nanti
akan dilakukan setelah data prediksi didapatkan dengan pc=2 karna ingin didapat
data 2 dimensi untuk bisa di-visualisasikan.

```python
# pc_1 = 8, pc_2 = 8, pc_3 = 8

pc = 8
pca_scale = PCA(pc, random_state=1)
dfready_scale_pca_1 = pca_scale.fit_transform(dfready_scale_1)
dfready_scale_pca_2 = pca_scale.fit_transform(dfready_scale_2)
dfready_scale_pca_3 = pca_scale.fit_transform(dfready_scale_3)

# dfready_scale_pca.shape
```

## Anomaly Detection

```python
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

### Model Training
Melakukan training data dengan model yang digunakan.

```python
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

Simpan ke dalam file untuk kebutuhan analisa lanjutan

```python
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

### Model Test

Fungsi decision_function untuk melihat score secara real, jika nilai score
anomali semakin minus maka semakin anomali.

```python
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

```python
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

```python
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

```python
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

```python
np.where(pred_rocov_1 == -1)
np.where(pred_ocsvm_1 == -1)
np.where(pred_isofor_1 == -1)
```

### Prediction

Prediksi dan dapatkan kira-kira data apa saja yang terindikasi fraudulent.
Berikut adalah gambaran hasil yang diberikan (dari data ready)

```python
data_predict_svm_1 = data_ready_1.iloc[idx_ocsvm_1[0],:]
data_predict_svm_2 = data_ready_2.iloc[idx_ocsvm_2[0],:]
data_predict_svm_3 = data_ready_3.iloc[idx_ocsvm_3[0],:]

dump(data_predict_svm_1, 'dataset/data_predict_svm_1.joblib')
dump(data_predict_svm_2, 'dataset/data_predict_svm_2.joblib')
dump(data_predict_svm_3, 'dataset/data_predict_svm_3.joblib')
```

```python
data_predict_cov_1 = data_ready_1.iloc[idx_rocov_1[0],:]
data_predict_cov_2 = data_ready_2.iloc[idx_rocov_2[0],:]
data_predict_cov_3 = data_ready_3.iloc[idx_rocov_3[0],:]

dump(data_predict_cov_1, 'dataset/data_predict_cov_1.joblib')
dump(data_predict_cov_2, 'dataset/data_predict_cov_2.joblib')
dump(data_predict_cov_3, 'dataset/data_predict_cov_3.joblib')
```

```python
data_predict_isofor_1 = data_ready_1.iloc[idx_isofor_1[0],:]
data_predict_isofor_2 = data_ready_2.iloc[idx_isofor_2[0],:]
data_predict_isofor_3 = data_ready_3.iloc[idx_isofor_3[0],:]

dump(data_predict_isofor_1, 'dataset/data_predict_isofor_1.joblib')
dump(data_predict_isofor_2, 'dataset/data_predict_isofor_2.joblib')
dump(data_predict_isofor_3, 'dataset/data_predict_isofor_3.joblib')
```

### 2d Visualization
Lalu interpretasikan (dan cocokkan) data-data yang anomali dari hasil reduksi
dimensi menggunakan visualisasi plotly berikut :

```python
df1 = pd.DataFrame(dfready_scale_pca_1,columns=['pc1','pc2','pc3','pc4','pc5','pc6','pc7','pc8'],index=data_ready_1.index)
df2 = pd.DataFrame(dfready_scale_pca_2,columns=['pc1','pc2','pc3','pc4','pc5','pc6','pc7','pc8'],index=data_ready_2.index)
df3 = pd.DataFrame(dfready_scale_pca_3,columns=['pc1','pc2','pc3','pc4','pc5','pc6','pc7','pc8'],index=data_ready_3.index)
```

```python
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

```python
df_tsne_1 = pd.DataFrame(dfready_pca_1,columns=['pc1','pc2'],index=data_ready_1.index)
df_tsne_2 = pd.DataFrame(dfready_pca_2,columns=['pc1','pc2'],index=data_ready_2.index)
df_tsne_3 = pd.DataFrame(dfready_pca_3,columns=['pc1','pc2'],index=data_ready_3.index)

# df_tsne_1 = pd.DataFrame(dfready_tsne_1,columns=['pc1','pc2'],index=data_ready_1.index)
# df_tsne_2 = pd.DataFrame(dfready_tsne_2,columns=['pc1','pc2'],index=data_ready_2.index)
# df_tsne_3 = pd.DataFrame(dfready_tsne_3,columns=['pc1','pc2'],index=data_ready_3.index)
```

```python
# Join data hasil scaling di atas dengan data_ready yang di atas.
df_1 = df_tsne_1.join(data_ready_1)
df_2 = df_tsne_2.join(data_ready_2)
df_3 = df_tsne_3.join(data_ready_3)
```

```python
df_pca_1 = df_tsne_1.copy()
df_pca_2 = df_tsne_2.copy()
df_pca_3 = df_tsne_3.copy()
```

```python
# Menggunakan Data Ready 1
df = df_1.copy()
fig = px.scatter(df, x="pc1", y="pc2", hover_data=[df.index.get_level_values(0), 
    df.index, 
    'pc1','pc2', 'total_call', 'total_duration'])
# fig.show()
plotly.offline.plot(fig, filename='assets/plt_data_ready_1.html', image='png', auto_open=True, output_type='file', image_width=800, image_height=600, validate=False)
```

Menggunakan data ready 1 maka akan menghasilkan plot seperti berikut :

![ ](assets/plt_data_ready_1.png)

```python
#Menggunakan Data Ready 2
df = df_2.copy()
fig = px.scatter(df, x="pc1", y="pc2", hover_data=[df.index.get_level_values(0), 
    df.index.get_level_values(1), 
    'pc1','pc2', 'total_call', 'total_duration'])
# fig.show()
plotly.offline.plot(fig, filename='assets/plt_data_ready_2.html', image='png', auto_open=True, output_type='file', image_width=800, image_height=600, validate=False)
```

Menggunakan data ready 2 maka akan menghasilkan plot seperti berikut :

![ ](assets/plt_data_ready_2.png)

```python
#Menggunakan Data Ready 3
df = df_3.copy()
fig = px.scatter(df, x="pc1", y="pc2", hover_data=[df.index.get_level_values(0), 
    df.index.get_level_values(1), 
    df.index.get_level_values(2),
    'pc1','pc2', 'total_call', 'total_duration'])
# fig.show()
plotly.offline.plot(fig, filename='assets/plt_data_ready_3.html', image='png', auto_open=True, output_type='file', image_width=800, image_height=600, validate=False)
```

Menggunakan data ready 3 maka akan menghasilkan plot seperti berikut :

![ ](assets/plt_data_ready_3.png)

Performa masing-masing model **Anomaly Detection** seperti gambar berikut :

```python
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

```python
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

```python
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

___

# Evaluasi

Bandingkan dengan dataframe **fraud** yang berupa data fraud berdasarkan rule
base untuk masing-masing hasil prediksi. Cek mana yang paling cocok atau paling
banyak menangkap nomor telepon fraud yang sama.

```python
fraud.set_index(['source_num'], inplace=True)
```

```python
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

```python
# Cek jumlah nomor telepon unik untuk masing-masing dataframe.
print(fraud.shape)
print(data_predict_svm_1.shape)
print(data_predict_cov_1.shape)
print(data_predict_isofor_1.shape)
```

```python
# Cek jumlah nomor telepon unik untuk masing-masing dataframe.
print(fraud.index.nunique())
print(data_predict_svm_1.index.get_level_values(0).nunique())
print(data_predict_cov_1.index.get_level_values(0).nunique())
print(data_predict_isofor_1.index.get_level_values(0).nunique())
```

```python
# Cek jumlah nomor telepon unik untuk masing-masing dataframe.
print(fraud.index.nunique())
print(data_predict_isofor_1.index.get_level_values(0).nunique())
print(data_predict_isofor_2.index.get_level_values(0).nunique())
print(data_predict_isofor_3.index.get_level_values(0).nunique())
```

Merge (union) data dari masing-masing hasil model prediksi untuk menyaring data
anomaly lebih banyak.

```python
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

Lanjutkan dengan melakukan komparasi (irisan) data nomor telepon asal yang unik
terhadap data fraud berdasarkan rule base.

```python
compare_svm = data_merge_svm.merge(fraud, how='inner', on=['source_num'])
compare_svm.index.nunique()
```

```python
compare_cov = data_merge_cov.merge(fraud, how='inner', on=['source_num'])
compare_cov.index.nunique()
```

```python
compare_isofor = data_merge_isofor.merge(fraud, how='inner', on=['source_num'])
compare_isofor.index.nunique()
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
