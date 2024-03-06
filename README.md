# Laporan Proyek Machine Learning - Muhammad Fathul Radhiansyah

## Domain Proyek

**Latar Belakang**

Dalam industri perjalanan dan pariwisata, asuransi perjalanan memiliki peran penting dalam memberikan perlindungan untuk berbagai risiko terkait perjalanan, seperti pembatalan perjalanan, keadaan darurat medis, kehilangan bagasi, dan penundaan penerbangan. Namun, tingkat konversi asuransi perjalanan seringkali rendah, sehingga memahami perilaku dan preferensi pelanggan menjadi kunci untuk meningkatkan penjualan asuransi perjalanan.

Proyek ini bertujuan untuk mengembangkan model prediksi asuransi perjalanan yang dapat memprediksi apakah pelanggan akan membeli asuransi perjalanan atau tidak berdasarkan berbagai fitur. Model ini akan membantu agen perjalanan dan perusahaan asuransi untuk mengidentifikasi pelanggan potensial, mempersonalisasi produk asuransi mereka, dan meningkatkan tingkat konversi.

**Mengapa dan Bagaimana Masalah Ini Harus Diselesaikan?**

Tingkat konversi asuransi perjalanan yang rendah menunjukkan perlunya pemahaman lebih dalam tentang perilaku pelanggan dan faktor-faktor yang memengaruhi keputusan pembelian mereka. Analisis terhadap fitur-fitur demografis dan terkait perjalanan pelanggan dapat memberikan wawasan yang berharga untuk meningkatkan efektivitas kampanye pemasaran dan penjualan asuransi perjalanan.

Namun, menganalisis fitur-fitur ini secara manual untuk setiap pelanggan dapat menjadi proses yang lambat dan rentan terhadap kesalahan. Oleh karena itu, pengembangan model prediksi yang otomatis dan akurat sangat diperlukan untuk mengidentifikasi pola-pola yang tersembunyi dalam data dan memprediksi kecenderungan pembelian pelanggan.

Referensi: [Predicting Travel Insurance Purchases in an Insurance Firm through Machine Learning Methods after COVID-19](https://www.researchgate.net/publication/373895975_Predicting_Travel_Insurance_Purchases_in_an_Insurance_Firm_through_Machine_Learning_Methods_after_COVID-19)

## Business Understanding

### Problem Statements
Berdasarkan kondisi yang telah diuraikan sebelumnya, perusahaan akan mengembangkan sebuah sistem prediksi asuransi perjalanan yang dapat memprediksi apakah pelanggan akan membeli asuransi perjalanan atau tidak berdasarkan berbagai fitur, untuk menjawab permasalahan berikut.

- Fitur mana yang memiliki dampak paling signifikan terhadap keputusan nasabah untuk membeli asuransi perjalanan?
- Membangun model prediktif dengan nilai ROC AUC di atas 70% yang dapat memprediksi dengan akurasi tinggi apakah pelanggan akan membeli asuransi perjalanan berdasarkan fitur-fitur yang telah diidentifikasi? Akurasi tinggi dalam konteks ini akan diukur menggunakan metrik evaluasi ROC AUC.

### Goals
Untuk menjawab pertanyaan tersebut, Anda akan membuat predictive modelling dengan tujuan atau goals sebagai berikut:

- Menggunakan metode .feature_importances_ dari model yang dipilih untuk menemukan fitur-fitur yang memiliki dampak paling signifikan terhadap keputusan pembelian asuransi perjalanan.
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

Statistika deskriptif untuk fitur kategorikal:
|                        | Employment Type          | GraduateOrNot   | FrequentFlyer   | EverTravelledAbroad   |
|------------------------|--------------------------|-----------------|-----------------|-----------------------|
| count                  | 1249                     | 1249            | 1249            | 1249                  |
| unique                 | 2                        | 2               | 2               | 2                     |
| top                    | Private Sector/Self Employed | Yes           | No              | No                    |
| freq                   | 876                      | 1047            | 954             | 1005                  |


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

![Histplot of Numerical Features](https://drive.google.com/uc?id=1pzWwJ_Jnihq3khEqAROKPhL8PBR14QgT)

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

**Pairplot of Numerical Features**

![Pairplot of Numerical Features](https://drive.google.com/uc?id=1dCJtGTJqnTpf3zszRxoLlaxiZlRju2rH)

Berdasarkan grafik, diperoleh bahwa tidak terdapat pola menarik yang menunjukkan hubungan dari TravelInsurance dengan fitur-fitur numerik lainnya.

**Heatmap Correlation of Numerical Features**

![Heatmap Correlation of Numerical Features](https://drive.google.com/uc?id=1_xgLdYvrE6MRLTifWpDod6kXumXVZ5zZ)

Dari matriks tersebut, dapat dilihat bahwa:

- Age: memiliki korelasi positif yang rendah dengan keputusan untuk membeli asuransi perjalanan (TravelInsurance).
- AnnualIncome: memiliki korelasi positif yang sedang dengan keputusan untuk membeli asuransi perjalanan.
- FamilyMembers: memiliki korelasi positif yang rendah dengan keputusan untuk membeli asuransi perjalanan.
- ChronicDiseases: memiliki korelasi negatif yang rendah dengan keputusan untuk membeli asuransi perjalanan.

## Data Preparation

## Persiapan Data

Proses persiapan data adalah langkah penting dalam pengembangan model prediktif. Berikut adalah langkah-langkah detail yang dilakukan dalam proses persiapan data:

- **Menghapus Data Duplikat**: Terdapat 738 baris data duplikat yang diidentifikasi dan dihapus untuk menghindari bias dalam model. Data duplikat dapat memengaruhi kinerja model dengan meningkatkan pentingnya observasi tertentu.
- **Encoding Data Kategorikal**: Fitur-fitur kategorikal diubah menjadi representasi numerik agar dapat digunakan dalam model pembelajaran mesin. Ini diperlukan karena sebagian besar algoritma pembelajaran mesin membutuhkan data input dalam bentuk numerik.
- **Pemisahan Data menjadi Set Pelatihan dan Pengujian**: Data dibagi menjadi set pelatihan dan pengujian dengan rasio 80:20. Set pelatihan digunakan untuk melatih model, sementara set pengujian digunakan untuk mengevaluasi kinerjanya. Ini membantu menilai seberapa baik model menggeneralisasi data baru yang tidak terlihat.
- **Oversampling dengan SMOTE**: Teknik Synthetic Minority Over-sampling Technique (SMOTE) digunakan untuk menyeimbangkan distribusi kelas dengan oversampling kelas minoritas. Teknik ini menghasilkan sampel sintetis untuk kelas minoritas untuk mengatasi masalah ketidakseimbangan kelas, yang dapat menyebabkan model yang bias.
- **Standarisasi Fitur Numerik**: Fitur-fitur numerik seperti Usia, Jumlah Anggota Keluarga, dan Pendapatan Tahunan distandarisasi untuk memiliki mean 0 dan standar deviasi 1. Standarisasi penting untuk algoritma yang mengandalkan metrik jarak atau gradien, karena memastikan bahwa semua fitur memberikan kontribusi yang sama dalam proses pembelajaran model.

## Modeling

Pada tahap pemodelan, digunakan beberapa algoritma sebagai berikut:

- **Logistic Regression**: Cocok untuk klasifikasi biner dengan interpretasi yang mudah, namun cenderung kurang fleksibel dalam menangani hubungan yang kompleks.
- **RandomForestClassifier**: Ensambel pohon keputusan yang mampu menangani data tidak terstruktur, namun kompleksitas modelnya lebih tinggi.
- **GradientBoostingClassifier**: Ensambel pohon keputusan yang memperbaiki kesalahan secara berurutan, dapat menangani data yang tidak teratur, namun cenderung overfit jika tidak diatur dengan baik.
- **AdaBoostClassifier**: Ensambel pohon keputusan yang memberikan bobot pada sampel yang salah diklasifikasi pada iterasi sebelumnya, cocok untuk menangani data tidak seimbang namun rentan terhadap noise dan outlier.

Model GradientBoostingClassifier dipilih sebagai model terbaik dengan matriks evaluasi yang lebih tinggi, terutama pada matriks ROC AUC dengan skor 77%, menggunakan teknik Cross Validation pada data training.

## Evaluation

Matriks evaluasi yang digunakan meliputi Accuracy, Precision, Recall, F1 Score, dan ROC AUC. ROC AUC dipilih sebagai matriks evaluasi utama karena kemampuannya dalam mengukur false positif dan false negatif, yang penting dalam kasus klasifikasi yang tidak seimbang.

Dengan memperhatikan matriks evaluasi ini, model klasifikasi terbaik dapat dipilih berdasarkan kinerjanya dalam memisahkan kelas positif dan negatif. Model dengan ROC AUC yang lebih tinggi dianggap lebih baik dalam memprediksi keputusan pembelian asuransi perjalanan.

![Comparison of evaluation model](https://drive.google.com/uc?id=1YkbUeUkemInxpR9Pm3v03cEVBcgxCrn7)  

Berdasarkan hasil evaluasi data training, model GradientBoostingClassifier terpilih sebagai model terbaik dengan matriks evaluasi yang lebih tinggi, terutama pada matriks ROC AUC dengan skor pengujian sebesar 75% dan skor pelatihan sebesar 77%. Analisis lebih lanjut menunjukkan bahwa fitur AnnualIncome, FamilyMembers, dan Age memiliki kontribusi signifikan dalam memprediksi keputusan pembelian asuransi perjalanan.
