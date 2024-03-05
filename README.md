# Laporan Proyek Machine Learning - Muhammad Fathul Radhiansyah

## Domain Proyek

**Latar Belakang**

Dalam industri perjalanan dan pariwisata, asuransi perjalanan merupakan komponen penting yang memberikan perlindungan untuk berbagai risiko yang terkait dengan perjalanan, seperti pembatalan perjalanan, keadaan darurat medis, kehilangan bagasi, dan penundaan penerbangan. Namun, tingkat konversi asuransi perjalanan bisa jadi rendah, dan memahami perilaku serta preferensi pelanggan sangat penting untuk meningkatkan penjualan asuransi perjalanan.

Proyek ini bertujuan untuk membangun model prediksi asuransi perjalanan yang dapat memprediksi apakah pelanggan akan membeli asuransi perjalanan atau tidak berdasarkan berbagai fitur. Model ini akan membantu agen perjalanan dan perusahaan asuransi untuk mengidentifikasi pelanggan potensial, mempersonalisasi produk asuransi mereka, dan meningkatkan tingkat konversi mereka.

**Mengapa dan bagaimana masalah tersebut harus diselesaikan?**

Tingkat konversi asuransi perjalanan relatif rendah, dan ada kebutuhan untuk memahami perilaku dan preferensi pelanggan untuk meningkatkan penjualan asuransi perjalanan. Menganalisis fitur demografis dan fitur terkait perjalanan nasabah, seperti usia, jenis kelamin, tujuan, durasi perjalanan, dan frekuensi perjalanan, dapat memberikan wawasan tentang kemungkinan mereka untuk membeli asuransi perjalanan.

Namun, menganalisis fitur-fitur ini secara manual untuk setiap pelanggan dapat memakan waktu dan rentan terhadap kesalahan. Oleh karena itu, diperlukan model prediksi otomatis dan akurat yang dapat menganalisis fitur-fitur ini dan memprediksi kemungkinan nasabah untuk membeli asuransi perjalanan.

Referensi: [Predicting Travel Insurance Purchases in an Insurance Firm through Machine Learning Methods after COVID-19](https://www.researchgate.net/publication/373895975_Predicting_Travel_Insurance_Purchases_in_an_Insurance_Firm_through_Machine_Learning_Methods_after_COVID-19)

## Business Understanding

### Problem Statements
Berdasarkan kondisi yang telah diuraikan sebelumnya, perusahaan akan mengembangkan sebuah sistem prediksi asuransi perjalanan yang dapat memprediksi apakah pelanggan akan membeli asuransi perjalanan atau tidak berdasarkan berbagai fitur, untuk menjawab permasalahan berikut.

- Fitur mana yang memiliki dampak paling signifikan terhadap keputusan nasabah untuk membeli asuransi perjalanan?
- Dapatkah kita memprediksi dengan akurasi tinggi apakah seorang nasabah akan membeli asuransi perjalanan berdasarkan fitur-fitur yang telah diidentifikasi?

### Goals
Untuk menjawab pertanyaan tersebut, Anda akan membuat predictive modelling dengan tujuan atau goals sebagai berikut:

- Mengidentifikasi fitur-fitur yang memiliki dampak paling signifikan terhadap keputusan nasabah untuk membeli asuransi perjalanan.
- Mengembangkan model pembelajaran mesin yang dapat memprediksi dengan akurasi tinggi apakah nasabah akan membeli asuransi perjalanan berdasarkan fitur-fitur yang telah diidentifikasi.

## Data Understanding
Data yang Anda gunakan pada proyek kali ini adalah "Travel Insurance Prediction Data" yang diunduh dari <a href="https://www.kaggle.com/datasets/tejashvi14/travel-insurance-prediction-data">Kaggle API</a>. Dataset ini memiliki 1987 baris dengan 9 fitur, yang terdiri fitur non-numerik seperti Employment Type, GraduateOrNot, FrequentFlyer, dan EverTravelledAbroad, serta fitur numerik seperti Age, AnnualIncome, FamilyMembers, dan ChronicDiseases. Kedelapan fitur ini adalah fitur yang akan Anda gunakan dalam menemukan pola pada data, sedangkan TravelInsurance merupakan fitur target.

### Berdasarkan informasi dari Kaggle, fitur-fitur pada Travel Insurance dataset adalah sebagai berikut:

- `Age` : Usia Pelanggan
- `Employment Type` : Sektor di Mana Pelanggan Bekerja
- `GraduateOrNot` : Apakah Pelanggan Lulus Kuliah atau Tidak
- `AnnualIncome` : Pendapatan Tahunan Pelanggan dalam Rupee India [Dibulatkan ke Nearest 50 Ribu Rupee]
- `FamilyMembers` : Jumlah Anggota dalam Keluarga Pelanggan
- `ChronicDiseases` : Apakah Pelanggan Menderita Penyakit atau Kondisi Mayor seperti Diabetes/Tekanan Darah Tinggi atau Asma, dll.
- `FrequentFlyer` : Data yang Didapat Berdasarkan Riwayat Pelanggan dalam Membeli Tiket Pesawat di Setidaknya 4 Kali Berbeda dalam 2 Tahun Terakhir [2017-2019].
- `EverTravelledAbroad` : Apakah Pelanggan Pernah Berpergian ke Luar Negeri [Tidak Necessarily Menggunakan Layanan Perusahaan]
- `Asuransi Perjalanan` : Apakah Pelanggan Membeli Paket Asuransi Perjalanan Selama Penawaran Pengenalan yang Diadakan pada Tahun 2019.

### Exploratory Data Analysis
- Melakukan beberapa tahapan yang diperlukan untuk memahami data, contohnya teknik visualisasi data atau exploratory data analysis.

## Data Preparation
Proses persiapan data sangat penting dalam pengembangan model prediktif. Berikut adalah beberapa langkah yang akan dilakukan dalam proses persiapan data:

- Drop Data Duplikat: Menghapus duplikat data akan membantu menghindari bias dalam model dan memastikan integritas data yang baik. Duplikat data dapat mempengaruhi pembelajaran model dengan memberikan bobot yang tidak proporsional terhadap sampel tertentu.
- Encoding Data Kategorikal: Data kategorikal perlu diubah menjadi representasi numerik agar dapat digunakan dalam model pembelajaran mesin. Ini dapat dilakukan dengan teknik seperti one-hot encoding atau label encoding, tergantung pada karakteristik data.
- Pemisahan Data Train dan Test: Data perlu dibagi menjadi set pelatihan dan pengujian. Set pelatihan digunakan untuk melatih model, sedangkan set pengujian digunakan untuk mengevaluasi kinerja model. Ini penting untuk mengukur seberapa baik model akan berkinerja pada data yang belum pernah dilihat sebelumnya.
- Oversampling dengan Metode SMOTE: Ketidakseimbangan kelas dalam data dapat mempengaruhi kinerja model. Metode SMOTE (Synthetic Minority Over-sampling Technique) digunakan untuk menyeimbangkan jumlah sampel antara kelas mayoritas dan minoritas dengan menciptakan sampel sintetis dari kelas minoritas.
- Standarisasi dengan StandarScaler pada Fitur Age, FamilyMembers, dan AnnualIncome: Standarisasi fitur numerik memastikan bahwa semua fitur memiliki skala yang serupa. Ini penting untuk algoritma yang sensitif terhadap skala, seperti regresi logistik atau SVM. Standarisasi juga membantu dalam konvergensi lebih cepat selama proses pembelajaran.

## Modeling
Tahapan ini membahas mengenai model machine learning yang digunakan untuk menyelesaikan permasalahan. Anda perlu menjelaskan tahapan dan parameter yang digunakan pada proses pemodelan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan kelebihan dan kekurangan dari setiap algoritma yang digunakan.
- Jika menggunakan satu algoritma pada solution statement, lakukan proses improvement terhadap model dengan hyperparameter tuning. **Jelaskan proses improvement yang dilakukan**.
- Jika menggunakan dua atau lebih algoritma pada solution statement, maka pilih model terbaik sebagai solusi. **Jelaskan mengapa memilih model tersebut sebagai model terbaik**.

## Evaluation
Pada bagian ini anda perlu menyebutkan metrik evaluasi yang digunakan. Lalu anda perlu menjelaskan hasil proyek berdasarkan metrik evaluasi yang digunakan.

Sebagai contoh, Anda memiih kasus klasifikasi dan menggunakan metrik **akurasi, precision, recall, dan F1 score**. Jelaskan mengenai beberapa hal berikut:
- Penjelasan mengenai metrik yang digunakan
- Menjelaskan hasil proyek berdasarkan metrik evaluasi

Ingatlah, metrik evaluasi yang digunakan harus sesuai dengan konteks data, problem statement, dan solusi yang diinginkan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan formula metrik dan bagaimana metrik tersebut bekerja.

**---Ini adalah bagian akhir laporan---**

_Catatan:_
- _Anda dapat menambahkan gambar, kode, atau tabel ke dalam laporan jika diperlukan. Temukan caranya pada contoh dokumen markdown di situs editor [Dillinger](https://dillinger.io/), [Github Guides: Mastering markdown](https://guides.github.com/features/mastering-markdown/), atau sumber lain di internet. Semangat!_
- Jika terdapat penjelasan yang harus menyertakan code snippet, tuliskan dengan sewajarnya. Tidak perlu menuliskan keseluruhan kode project, cukup bagian yang ingin dijelaskan saja.
