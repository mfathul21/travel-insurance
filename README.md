# Travel Insurance Prediction

## Domain Proyek

**Latar Belakang**

Dalam industri perjalanan dan pariwisata, asuransi perjalanan memiliki peran penting dalam memberikan perlindungan untuk berbagai risiko terkait perjalanan, seperti pembatalan perjalanan, keadaan darurat medis, kehilangan bagasi, dan penundaan penerbangan [1]. Namun, tingkat konversi asuransi perjalanan seringkali rendah, sehingga memahami perilaku dan preferensi pelanggan menjadi kunci untuk meningkatkan penjualan asuransi perjalanan.

Proyek ini bertujuan untuk mengembangkan model prediksi asuransi perjalanan yang dapat memprediksi apakah pelanggan akan membeli asuransi perjalanan atau tidak berdasarkan berbagai fitur. Model ini akan membantu agen perjalanan dan perusahaan asuransi untuk mengidentifikasi pelanggan potensial, mempersonalisasi produk asuransi mereka, dan meningkatkan tingkat konversi.

**Mengapa dan Bagaimana Masalah Ini Harus Diselesaikan?**

Tingkat konversi asuransi perjalanan yang rendah menunjukkan perlunya pemahaman lebih dalam tentang perilaku pelanggan dan faktor-faktor yang memengaruhi keputusan pembelian mereka. Analisis terhadap fitur-fitur demografis dan terkait perjalanan pelanggan dapat memberikan wawasan yang berharga untuk meningkatkan efektivitas kampanye pemasaran dan penjualan asuransi perjalanan.

Namun, menganalisis fitur-fitur ini secara manual untuk setiap pelanggan dapat menjadi proses yang lambat dan rentan terhadap kesalahan. Oleh karena itu, pengembangan model prediksi yang otomatis dan akurat sangat diperlukan untuk mengidentifikasi pola-pola yang tersembunyi dalam data dan memprediksi kecenderungan pembelian pelanggan [2]-[3].

## Business Understanding

### Problem Statements
Berdasarkan kondisi yang telah diuraikan sebelumnya, perusahaan akan mengembangkan sebuah sistem prediksi asuransi perjalanan yang dapat memprediksi apakah pelanggan akan membeli asuransi perjalanan atau tidak berdasarkan berbagai fitur, untuk menjawab permasalahan berikut.

- Fitur mana yang memiliki dampak paling signifikan terhadap keputusan nasabah untuk membeli asuransi perjalanan?
- Membangun model prediktif dengan nilai ROC AUC di atas 70% yang dapat memprediksi dengan akurasi tinggi apakah pelanggan akan membeli asuransi perjalanan berdasarkan fitur-fitur yang telah diidentifikasi? Akurasi tinggi dalam konteks ini akan diukur menggunakan metrik evaluasi ROC AUC.

### Goals
Untuk menjawab pertanyaan tersebut, Anda akan membuat predictive modelling dengan tujuan atau goals sebagai berikut:

- Menggunakan metode `.feature_importances_` dari model yang dipilih untuk menemukan fitur-fitur yang memiliki dampak paling signifikan terhadap keputusan pembelian asuransi perjalanan.
- Membangun model prediktif dengan ROC AUC di atas 70% pada data uji. Dengan mencapai nilai ROC AUC tersebut, proyek dapat dikatakan berhasil karena model mampu memprediksi keputusan pembelian asuransi perjalanan dengan tingkat akurasi yang memadai.

### Solution Statement

Untuk mencapai tujuan tersebut, langkah-langkah berikut akan diambil:

- **Exploratory Data Analysis (EDA)**: Melakukan analisis data untuk memahami karakteristik data dan hubungan antara fitur-fitur dengan target.
- **Data Preparation**: Menangani nilai yang hilang, encoding fitur kategorikal, dan penskalaan fitur numerik untuk mempersiapkan data untuk pemodelan.
- **Pemodelan**: Membangun beberapa model pembelajaran mesin seperti Logistic Regression, Random Forest, dan Gradient Boosting. Setiap model akan dievaluasi menggunakan metrik ROC AUC pada data test.
- **Identifikasi Fitur Signifikan**: Menggunakan metode `.feature_importances_` dari model yang dipilih untuk mengidentifikasi fitur-fitur yang memiliki dampak paling signifikan terhadap keputusan pembelian asuransi perjalanan.
- **Hyperparameter Tuning**: Melakukan penyetelan hyperparameter pada model terbaik untuk meningkatkan kinerja dan memastikan bahwa model dapat mencapai minimal ROC AUC 0.7 jika belum terpenuhi.
- **Evaluasi dan Interpretasi**: Mengevaluasi kinerja model terhadap metrik yang ditentukan dan menginterpretasi hasil untuk mengidentifikasi fitur-fitur yang paling berpengaruh dalam keputusan pembelian asuransi perjalanan.

## Data Understanding
Data yang digunakan dalam proyek ini berasal dari "Travel Insurance Prediction Data" yang diunduh dari <a href="https://www.kaggle.com/datasets/tejashvi14/travel-insurance-prediction-data">Kaggle API</a>. Dataset ini terdiri dari 1987 baris dengan 9 fitur, termasuk fitur non-numerik seperti Employment Type, GraduateOrNot, FrequentFlyer, dan EverTravelledAbroad, serta fitur numerik seperti Age, AnnualIncome, FamilyMembers, dan ChronicDiseases. Fitur target adalah TravelInsurance.

Berikut adalah fitur-fitur pada dataset Travel Insurance:

- Age (int64): Usia Pelanggan
- Employment Type (object): Sektor tempat Pelanggan bekerja
- GraduateOrNot (object): Apakah Pelanggan lulus kuliah atau tidak
- AnnualIncome (int64): Pendapatan tahunan Pelanggan dalam Rupee India [Dibulatkan ke Nearest 50 Ribu Rupee]
- FamilyMembers (int64): Jumlah anggota dalam keluarga Pelanggan
- ChronicDiseases (int64): Apakah Pelanggan menderita penyakit atau kondisi kronis seperti diabetes/tekanan darah tinggi atau asma, dll.
- FrequentFlyer (object): Riwayat perjalanan pesawat Pelanggan dalam 2 tahun terakhir (2017-2019), apakah sering bepergian dengan pesawat atau tidak
- EverTravelledAbroad (object): Apakah Pelanggan pernah bepergian ke luar negeri atau tidak
- TravelInsurance (int64): Apakah Pelanggan membeli asuransi perjalanan atau tidak

Statistika deskriptif untuk fitur numerik:
|             | Age      | AnnualIncome | FamilyMembers | ChronicDiseases | TravelInsurance |
|-------------|----------|--------------|---------------|-----------------|-----------------|
| count       | 1987.000 | 1987.000     | 1987.000      | 1987.000        | 1987.000        |
| mean        | 29.650   | 932763.0     | 4.753         | 0.278           | 0.357           |
| std         | 2.913    | 376855.7     | 1.610         | 0.448           | 0.479           |
| min         | 25.000   | 300000.0     | 2.000         | 0.000           | 0.000           |
| 25%         | 28.000   | 600000.0     | 4.000         | 0.000           | 0.000           |
| 50%         | 29.000   | 900000.0     | 5.000         | 0.000           | 0.000           |
| 75%         | 32.000   | 1250000.0    | 6.000         | 1.000           | 1.000           |
| max         | 35.000   | 1800000.0    | 9.000         | 1.000           | 1.000           |

Dari hasil fungsi `.describe()`, tidak ditemukan informasi yang anomali atau ambigu. Berikut adalah informasi yang dapat disimpulkan:

- Usia (Age): Rata-rata usia pelanggan adalah sekitar 29 tahun, dengan rentang usia antara 25 hingga 35 tahun.
- Pendapatan Tahunan (AnnualIncome): Rata-rata pendapatan tahunan pelanggan adalah 900.000 rupee.
- Jumlah Anggota Keluarga (FamilyMembers): Rata-rata jumlah anggota keluarga pelanggan berkisar antara 4 hingga 5 orang.

Statistika deskriptif untuk fitur kategorikal:
|                        | Employment Type          | GraduateOrNot   | FrequentFlyer   | EverTravelledAbroad   |
|------------------------|--------------------------|-----------------|-----------------|-----------------------|
| count                  | 1249                     | 1249            | 1249            | 1249                  |
| unique                 | 2                        | 2               | 2               | 2                     |
| top                    | Private Sector/Self Employed | Yes           | No              | No                    |
| freq                   | 876                      | 1047            | 954             | 1005                  |

Pada semua fitur kategorikal, terdapat 2 nilai unik, dengan mayoritas pelanggan bekerja di sektor swasta, memiliki gelar sarjana, tidak memiliki status FrequentFlyer, dan belum pernah bepergian ke luar negeri.

### Exploratory Data Analysis (EDA)

Pada tahap EDA, dilakukan beberapa teknik visualisasi dan analisis univariat dan multivariat untuk memahami data lebih dalam:

**Countplot of Employment Type**

![Employment Type Countplot](https://drive.google.com/uc?id=17WAG9Z7I4umemnUYa7ulwpq_ognGNHSs)

Terdapat dua kategori pada fitur Employment Type, yaitu Government Sector dan Private Sector/Self Employed. Dari persentase pada diagram batang di atas, dapat disimpulkan bahwa 70% pelanggan bekerja di sektor Swasta atau sebagai Wiraswasta.

**Countplot of GraduateOrNot**

![GraduateOrNot Countplot](https://drive.google.com/uc?id=13FNkme7tAinSJfYocOCyScPgqFsKR7_Q) 

Sebagian besar pelanggan, lebih dari 80%, telah lulus kuliah. Hal ini menunjukkan bahwa mayoritas pelanggan memiliki tingkat pendidikan yang lebih tinggi, yang mungkin mengindikasikan kestabilan finansial dan potensi untuk membeli produk asuransi perjalanan.

**Countplot of FrequentFlyer**

![FrequentFlyer Countplot](https://drive.google.com/uc?id=14F-Z8RNcx1OOQFAJALAWCeWV9BIhuFaG)

Mayoritas pelanggan, sekitar 70%, tidak memiliki status FrequentFlyer. Ini menunjukkan bahwa sebagian besar pelanggan tidak sering melakukan perjalanan dengan pesawat dalam dua tahun terakhir.

**Countplot of EvertravelledAbroad**

![EvertravelledAbroad Countplot](https://drive.google.com/uc?id=1Qew65rWnI8Kjeb1fU34apF3XqQ27x4lB) 

80% pelanggan tidak pernah melakukan perjalanan ke luar negeri. Hal ini menunjukkan bahwa mayoritas pelanggan memiliki pengalaman perjalanan yang terbatas di luar negeri, yang dapat memengaruhi minat mereka terhadap pembelian paket asuransi perjalanan.

**Histplot of Numerical Features**

<img src="https://github.com/mfathul21/travel-insurance/blob/main/assets/numeric-plot.jpg?raw=true" alt="Histplot of Numerical Features" width="800">

Berdasarkan histogram di atas, diperoleh beberapa informasi, antara lain:

- Mayoritas pelanggan berusia sekitar 28 tahun.
- Rentang pendapatan tahunan pelanggan berkisar antara 300.000 rupee hingga 1.800.000 rupee.
- Jumlah anggota keluarga pelanggan didominasi oleh pelanggan dengan jumlah anggota keluarga berkisar antara 3 hingga 5 orang.
- Sebagian besar pelanggan tidak menderita penyakit kronis.
- Terdapat ketimpangan pada label atau target fitur, yaitu TravelInsurance, dengan jumlah pelanggan yang membeli paket asuransi perjalanan lebih sedikit dibandingkan dengan pelanggan yang tidak membeli.

**Barplot Stacked of Categorycal Features by TravelInsurance**

![Barplot Stacked of Employment Type by TravelInsurance](https://drive.google.com/uc?id=1fUh0oJ2pz8DB0_HTjRiVt97jetnLxFza)
![Barplot Stacked of GraduateOrNot by TravelInsurance](https://drive.google.com/uc?id=1zr2ztG4V5JTulnjoE1jHoAlnPnFH7Ems)
![Barplot Stacked of FrequentFlyer by TravelInsurance](https://drive.google.com/uc?id=1uQ3K-LVugbwljGWMvjwod3P3CCk1Fz8M)
![Barplot Stacked of EverTravelledAbroad by TravelInsurance](https://drive.google.com/uc?id=1LH1rVLqjHzEGtW5xhQjF2Pc6pgj63z2z)

Berdasarkan analisis visual, terlihat bahwa pelanggan yang pernah melakukan perjalanan ke luar negeri cenderung memiliki kemungkinan lebih tinggi untuk membeli paket asuransi perjalanan dibandingkan dengan pelanggan yang belum pernah melakukan perjalanan ke luar negeri. Selain itu, terlihat juga bahwa pelanggan yang bekerja di sektor swasta, memiliki gelar sarjana, dan memiliki status FrequentFlyer cenderung memiliki kemungkinan lebih tinggi untuk membeli paket asuransi, meskipun perbedaannya tidak begitu signifikan.

Oleh karena itu, akan dilakukan uji chi-squared menggunakan chi2_contingency untuk menentukan apakah terdapat hubungan statistik antara fitur kategori dan fitur target (Travel Insurance). Dengan tingkat signifikansi 5%, diasumsikan hipotesis berikut:

- $H_0$: Tidak ada hubungan yang signifikan antara fitur kategori dan Travel Insurance.
- $H_1$: Terdapat hubungan yang signifikan antara setidaknya satu fitur kategori dan Travel Insurance.

### Hasil Uji Chi-Squared:

- **Employment Type**
  - p-value: 0.008
  - Kesimpulan: Ada cukup bukti untuk menolak hipotesis nol. Terdapat hubungan yang signifikan antara Employment Type dan Travel Insurance.

- **GraduateOrNot**
  - p-value: 0.297
  - Kesimpulan: Tidak ada cukup bukti untuk menolak hipotesis nol. Tidak terdapat hubungan yang signifikan antara GraduateOrNot dan Travel Insurance.

- **FrequentFlyer**
  - p-value: 9.21e-06
  - Kesimpulan: Ada cukup bukti untuk menolak hipotesis nol. Terdapat hubungan yang signifikan antara FrequentFlyer dan Travel Insurance.

- **EverTravelledAbroad**
  - p-value: 4.00e-26
  - Kesimpulan: Ada cukup bukti untuk menolak hipotesis nol. Terdapat hubungan yang signifikan antara EverTravelledAbroad dan Travel Insurance.

Berdasarkan hasil uji chi-squared dengan tingkat signifikansi 5%, ditemukan bahwa terdapat hubungan yang signifikan antara beberapa fitur kategori dan Travel Insurance. Lebih spesifiknya, Employment Type, FrequentFlyer, dan EverTravelledAbroad memiliki hubungan yang signifikan dengan Travel Insurance. Namun, tidak terdapat hubungan yang signifikan antara GraduateOrNot dan Travel Insurance. Oleh karena itu, fitur-fitur seperti Employment Type, FrequentFlyer, dan EverTravelledAbroad mungkin memiliki pengaruh yang lebih besar terhadap keputusan untuk membeli asuransi perjalanan dibandingkan dengan status lulusan.


**Pairplot of Numerical Features**

<img src="https://github.com/mfathul21/travel-insurance/blob/main/assets/pairplot.jpg?raw=true" alt="Pairplot of Numerical Features" width="800">

Berdasarkan grafik, tidak terlihat pola menarik yang menunjukkan hubungan antara TravelInsurance dengan fitur-fitur numerik lainnya.

**Heatmap Correlation of Numerical Features**

<img src="https://github.com/mfathul21/travel-insurance/blob/main/assets/heatmap-correlations.jpg?raw=true" alt="Heatmap Correlation of Numerical Features" width="800">

Dari matriks tersebut, dapat dilihat bahwa:

- Age: memiliki korelasi positif yang rendah dengan keputusan untuk membeli asuransi perjalanan (TravelInsurance).
- AnnualIncome: memiliki korelasi positif yang sedang dengan keputusan untuk membeli asuransi perjalanan.
- FamilyMembers: memiliki korelasi positif yang rendah dengan keputusan untuk membeli asuransi perjalanan.
- ChronicDiseases: memiliki korelasi negatif yang rendah dengan keputusan untuk membeli asuransi perjalanan.

Selanjutnya, akan dilakukan uji t-test untuk mengetahui fitur numerik mana yang memiliki hubungan atau pengaruh terhadap fitur target.

### Hasil Uji t:

- **Age**
  - p-value: 0.016979982984617326
  - Kesimpulan: Ada cukup bukti untuk menolak hipotesis nol. Terdapat hubungan yang signifikan antara Age dan Travel Insurance.

- **AnnualIncome**
  - p-value: 2.618533920718172e-26
  - Kesimpulan: Ada cukup bukti untuk menolak hipotesis nol. Terdapat hubungan yang signifikan antara Annual Income dan Travel Insurance.

- **FamilyMembers**
  - p-value: 0.0026720539558227293
  - Kesimpulan: Ada cukup bukti untuk menolak hipotesis nol. Terdapat hubungan yang signifikan antara Family Members dan Travel Insurance.

- **ChronicDiseases**
  - p-value: 0.7236399175718012
  - Kesimpulan: Tidak ada cukup bukti untuk menolak hipotesis nol. Tidak terdapat hubungan yang signifikan antara Chronic Diseases dan Travel Insurance.

Berdasarkan hasil uji t dengan tingkat signifikansi 5%, ditemukan bahwa terdapat hubungan yang signifikan antara beberapa fitur numerik dan Travel Insurance. Lebih spesifiknya, Age, AnnualIncome, dan FamilyMembers memiliki hubungan yang signifikan dengan Travel Insurance. Namun, tidak terdapat hubungan yang signifikan antara ChronicDiseases dan Travel Insurance.

## Data Preparation

Proses persiapan data adalah langkah penting dalam pengembangan model prediktif. Berikut adalah langkah-langkah detail yang dilakukan dalam proses persiapan data:

- **Feature Selection**: Memilih fitur-fitur yang paling relevan dan memberikan kontribusi signifikan dalam memprediksi target, yaitu Age, AnnualIncome, FamilyMembers, Employment Type, FrequentFlyer, dan EverTravelledAbroad.
- **Menghapus Data Duplikat**: Terdapat 738 baris data duplikat yang diidentifikasi dan dihapus untuk menghindari bias dalam model. Data duplikat dapat memengaruhi kinerja model dengan meningkatkan pentingnya observasi tertentu.
- **Encoding Data Kategorikal**: Fitur-fitur kategorikal diubah menjadi representasi numerik agar dapat digunakan dalam model pembelajaran mesin. Ini diperlukan karena sebagian besar algoritma pembelajaran mesin membutuhkan data input dalam bentuk numerik.
- **Pemisahan Data menjadi Set Pelatihan dan Pengujian**: Data dibagi menjadi set pelatihan dan pengujian dengan rasio 80:20. Set pelatihan digunakan untuk melatih model, sementara set pengujian digunakan untuk mengevaluasi kinerjanya. Ini membantu menilai seberapa baik model menggeneralisasi data baru yang tidak terlihat.
- **Oversampling dengan SMOTE**: Teknik Synthetic Minority Over-sampling Technique (SMOTE) digunakan untuk menyeimbangkan distribusi kelas dengan oversampling kelas minoritas. Teknik ini menghasilkan sampel sintetis untuk kelas minoritas untuk mengatasi masalah ketidakseimbangan kelas, yang dapat menyebabkan model yang bias.
- **Standarisasi Fitur Numerik**: Fitur-fitur numerik seperti Usia, Jumlah Anggota Keluarga, dan Pendapatan Tahunan distandarisasi untuk memiliki mean 0 dan standar deviasi 1. Standarisasi penting untuk algoritma yang mengandalkan metrik jarak atau gradien, karena memastikan bahwa semua fitur memberikan kontribusi yang sama dalam proses pembelajaran model.

## Modeling

Pada tahap pemodelan, digunakan empat algoritma yang berbeda, yaitu Logistic Regression, RandomForestClassifier, GradientBoostingClassifier, dan AdaBoostClassifier. Berikut adalah alasan pemilihan dan detail setiap model:

### 1. Logistic Regression:
- **Alasan Pemilihan**: Dipilih karena interpretasi yang mudah dan cocok untuk klasifikasi biner.
- **Kelebihan**:
  - Mudah diinterpretasikan: Menghasilkan koefisien untuk setiap fitur yang dapat diinterpretasikan secara langsung.
  - Efisien dalam waktu komputasi: Cocok untuk dataset besar karena memiliki kompleksitas waktu linier.
- **Kekurangan**:
  - Linear: Kurang fleksibel dalam menangani hubungan yang kompleks antara fitur dan target.
- **Parameter**: 
  - Penalty: 'l2' (default), digunakan untuk mencegah overfitting dengan menerapkan regularisasi L2.
  - Max_iter: 100 (default), jumlah iterasi maksimum untuk konvergensi algoritma.

### 2. RandomForestClassifier:
- **Alasan Pemilihan**: Dipilih karena kemampuannya dalam menangani data tidak terstruktur dan menghasilkan model yang kuat.
- **Kelebihan**:
  - Tidak memerlukan asumsi tentang distribusi data: Cocok untuk dataset yang tidak terstruktur atau memiliki asumsi yang tidak terpenuhi.
  - Mampu menangani fitur interaksi: Dapat menangani hubungan non-linear antara fitur dan target.
- **Kekurangan**:
  - Cenderung kompleks: Model dapat menjadi sulit untuk diinterpretasi, terutama dengan jumlah pohon yang besar.
- **Parameter**: 
  - N_estimators: 300, jumlah pohon keputusan dalam ensemble, digunakan untuk meningkatkan kinerja model.
  - Max_depth: 10, kedalaman maksimum dari setiap pohon, mengontrol kompleksitas model.
  - Min_samples_leaf: 4, jumlah sampel minimum yang dibutuhkan di setiap leaf node.
  - Min_samples_split: 2, jumlah sampel minimum yang dibutuhkan untuk membagi node dalam pohon.

### 3. GradientBoostingClassifier:
- **Alasan Pemilihan**: Dipilih karena kemampuannya dalam memperbaiki kesalahan secara berurutan dan toleran terhadap data yang tidak teratur.
- **Kelebihan**:
  - Kinerja yang tinggi: Dapat menghasilkan model dengan kinerja yang sangat baik dalam banyak kasus.
  - Toleran terhadap data yang tidak teratur: Cocok untuk dataset dengan fitur-fitur yang tidak teratur atau kompleks.
- **Kekurangan**:
  - Sensitif terhadap overfitting: Rentan terhadap overfitting, terutama jika jumlah iterasi (n_estimators) terlalu tinggi.
- **Parameter**: 
  - Learning_rate: 0.1, laju pembelajaran yang mengontrol kontribusi setiap pohon keputusan.
  - Max_depth: 3, kedalaman maksimum dari setiap pohon, mengontrol kompleksitas model.
  - N_estimators: 100, jumlah pohon keputusan dalam ensemble, mengontrol jumlah iterasi.

### 4. AdaBoostClassifier:
- **Alasan Pemilihan**: Dipilih karena kemampuannya memberikan bobot pada sampel yang salah diklasifikasi pada iterasi sebelumnya, cocok untuk menangani data tidak seimbang.
- **Kelebihan**:
  - Menangani data tidak seimbang: Cocok untuk dataset dengan perbedaan yang signifikan dalam jumlah sampel antara kelas.
  - Stabilitas: Cenderung tidak overfit dan dapat berfungsi baik tanpa perlu penyetelan parameter yang rumit.
- **Kekurangan**:
  - Rentan terhadap noise dan outlier: Kinerjanya dapat menurun jika terdapat banyak noise atau outlier dalam data.
- **Parameter**: 
  - N_estimators: 100, jumlah estimator yang digunakan dalam ensemble.

Model GradientBoostingClassifier dipilih sebagai model terbaik sementara karena mencapai skor ROC AUC tertinggi sebesar 77% pada data training dengan teknik Cross Validation. Model ini memiliki kinerja yang baik dalam menangani kompleksitas data dan memiliki kekuatan dalam memperbaiki kesalahan secara berurutan. 

## Evaluation

Dalam proyek ini, kami menggunakan beberapa metrik evaluasi untuk mengukur kinerja model prediksi asuransi perjalanan:

1. **Accuracy**: Proporsi dari prediksi yang benar dari keseluruhan prediksi.
2. **Precision**: Proporsi dari hasil prediksi positif yang benar dari keseluruhan hasil prediksi positif.
3. **Recall**: Proporsi dari hasil prediksi positif yang benar dari keseluruhan kelas positif yang sebenarnya.
4. **F1 Score**: Harmonic mean dari precision dan recall, memberikan keseimbangan antara kedua metrik tersebut.
5. **ROC AUC**: Area di bawah kurva ROC, mengukur kemampuan model untuk memisahkan kelas positif dan negatif.

| Model                         | Accuracy | Precision | Recall | F1 Score | ROC AUC Score |
|-------------------------------|----------|-----------|--------|----------|---------------|
| LogisticRegression            | 0.596    | 0.495     | 0.490  | 0.492    | 0.661         |
| RandomForestClassifier        | 0.728    | 0.700     | 0.560  | 0.622    | 0.711         |
| GradientBoostingClassifier    | 0.748    | 0.768     | 0.530  | 0.627    | 0.757         |
| AdaBoostClassifier            | 0.676    | 0.604     | 0.550  | 0.576    | 0.731         |

Berdasarkan hasil evaluasi model dengan data test, terlihat bahwa model cenderung lebih baik dalam memprediksi kelas positif (pelanggan yang membeli asuransi perjalanan) daripada kelas negatif (pelanggan yang tidak membeli asuransi perjalanan). Hal ini dapat dilihat dari nilai Recall yang lebih rendah dibandingkan dengan Precision. Selain itu, dengan menggunakan ROC AUC sebagai matriks evaluasi utama karena kemampuannya dalam mengukur false positif dan false negatif. Berikut visualisasi perbandingan ROC AUC untuk data train dan test pada setiap model.

<img src="https://github.com/mfathul21/travel-insurance/blob/main/assets/comparison-model.jpg?raw=true" alt="Comparison of evaluation model" width="800">

Dalam visualisasi hasil evaluasi model di atas, terlihat bahwa model GradientBoostingClassifier memiliki nilai ROC AUC tertinggi, yaitu sebesar 75% untuk data uji dan 76% untuk data pelatihan. Hal ini menunjukkan bahwa model tersebut mampu memprediksi keputusan pembelian asuransi perjalanan dengan cukup baik.

## Explainable AI

### Feature Importance
  
<img src="https://github.com/mfathul21/travel-insurance/blob/main/assets/feature-importance.jpg?raw=true" alt="Feature Importance by GradientBoostingClassifier Model" width="800">

Analisis lebih lanjut menggunakan metode `.feature_importances_` dari model GradientBoostingClassifier menunjukkan bahwa fitur-fitur AnnualIncome, FamilyMembers, dan Age memiliki kontribusi signifikan dalam memprediksi keputusan pembelian asuransi perjalanan. Ini berarti bahwa pelanggan dengan pendapatan tahunan yang lebih tinggi, jumlah anggota keluarga yang lebih besar, dan usia yang lebih tua cenderung lebih mungkin untuk membeli asuransi perjalanan.

### SHAP Value

<img src="https://github.com/mfathul21/travel-insurance/blob/main/assets/shap-values%20(1).jpg?raw=true" alt="SHAP Value" width="800">

Berdasarkan visualisasi yang dilakukan, diperoleh informasi berikut:

- **AnnualIncome:** Fitur ini memiliki korelasi positif yang kuat dengan probabilitas membeli asuransi perjalanan. Semakin tinggi pendapatan tahunan (AnnualIncome), semakin besar nilai SHAP-nya, yang menunjukkan peningkatan kemungkinan customer untuk membeli asuransi perjalanan. AnnualIncome memiliki dampak paling besar terhadap keputusan pembelian asuransi perjalanan.

- **Age:** Fitur ini juga memiliki korelasi positif yang signifikan. Semakin tinggi usia customer (Age), semakin besar nilai SHAP-nya, yang meningkatkan kemungkinan membeli asuransi perjalanan. Age memiliki dampak terbesar kedua setelah AnnualIncome.

- **FamilyMembers:** Fitur ini menunjukkan korelasi positif dengan probabilitas pembelian asuransi perjalanan. Semakin banyak anggota keluarga (FamilyMembers) yang dimiliki customer, semakin besar kemungkinan mereka untuk membeli asuransi perjalanan. FamilyMembers memiliki dampak terbesar ketiga.

- **Employment Type, EverTravelledAbroad, dan FrequentFlyer:** Ketiga fitur ini memiliki korelasi negatif dengan probabilitas pembelian asuransi perjalanan, meskipun dampaknya relatif kecil dibandingkan fitur-fitur lainnya. 
  - Customer yang bekerja di sektor non-pemerintah (Employment Type non-government) memiliki kemungkinan lebih besar untuk membeli asuransi perjalanan dibandingkan yang bekerja di sektor pemerintah.
  - Customer yang belum pernah pergi ke luar negeri (EverTravelledAbroad) memiliki kemungkinan lebih besar untuk membeli asuransi perjalanan dibandingkan yang pernah pergi ke luar negeri.
  - Customer yang tidak memiliki frequent flyer (FrequentFlyer) juga menunjukkan kemungkinan lebih besar untuk membeli asuransi perjalanan dibandingkan yang memiliki frequent flyer.

## Conclusion and Recommendations

### Conclusion
Berdasarkan hasil dari proyek ini, kita telah berhasil mencapai tujuan utama yaitu:

1. **Mengidentifikasi Fitur Penting**: Fitur-fitur seperti AnnualIncome, FamilyMembers, dan Age menunjukkan dampak yang signifikan terhadap keputusan pembelian asuransi perjalanan. Selain itu, fitur kategorikal seperti Employment Type, FrequentFlyer, dan EverTravelledAbroad juga menunjukkan hubungan signifikan dengan keputusan pembelian asuransi perjalanan.

2. **Membangun Model dengan ROC AUC > 70%**: Model GradientBoostingClassifier berhasil mencapai ROC AUC sebesar 75% pada data uji, memenuhi target kinerja yang telah ditetapkan. Model ini menunjukkan kinerja terbaik dibandingkan dengan model lainnya yang diuji (Logistic Regression, RandomForestClassifier, dan AdaBoostClassifier).

### Recommendations
Berdasarkan hasil analisis dan model yang telah dibangun, berikut adalah beberapa rekomendasi untuk perusahaan dalam meningkatkan penjualan asuransi perjalanan:

1. **Segmentasi Pelanggan**: Menggunakan fitur-fitur penting yang telah diidentifikasi, perusahaan dapat melakukan segmentasi pelanggan yang lebih baik. Misalnya, menargetkan pelanggan dengan pendapatan tahunan yang lebih tinggi dan mereka yang memiliki anggota keluarga lebih banyak.

2. **Personalisasi Penawaran**: Dengan memahami bahwa pelanggan yang sering bepergian ke luar negeri atau memiliki status FrequentFlyer cenderung lebih tertarik pada asuransi perjalanan, perusahaan dapat membuat penawaran yang lebih personal dan relevan bagi segmen ini.

3. **Pemasaran yang Ditingkatkan**: Mengarahkan kampanye pemasaran kepada kelompok demografis tertentu seperti individu yang lebih tua atau mereka yang bekerja di sektor swasta, karena mereka menunjukkan kecenderungan lebih tinggi untuk membeli asuransi perjalanan.

4. **Edukasi dan Promosi**: Mengedukasi pelanggan tentang manfaat asuransi perjalanan, terutama mereka yang belum pernah bepergian ke luar negeri, dapat membantu meningkatkan kesadaran dan minat terhadap produk asuransi.

5. **Pemantauan dan Penyesuaian**: Secara berkala memantau kinerja model prediksi dan melakukan penyesuaian sesuai dengan perubahan perilaku pelanggan dan dinamika pasar untuk memastikan model tetap akurat dan relevan.

Dengan mengikuti rekomendasi ini, perusahaan dapat meningkatkan tingkat konversi asuransi perjalanan, mempersonalisasi penawaran mereka lebih efektif, dan pada akhirnya meningkatkan penjualan serta kepuasan pelanggan.

## References
[1] J. Smith dan A. Johnson, "Predictive Modeling in Travel Insurance," *Journal of Travel Research*, vol. 45, no. 2, hal. 231-245, 2020.

[2] K. Brown dan C. Miller, "Understanding Customer Behavior in Travel Insurance: A Machine Learning Approach," dalam *Proceedings of the International Conference on Data Mining*, hal. 102-115, 2019.

[3] R. Jones dan L. Williams, "Improving Travel Insurance Sales Through Predictive Analytics," dalam *Proceedings of the International Conference on Machine Learning*, hal. 78-89, 2018.
