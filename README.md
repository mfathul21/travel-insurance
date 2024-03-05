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

### Exploratory Data Analysis (EDA)

Pada tahap EDA, dilakukan beberapa teknik visualisasi dan analisis univariat dan multivariat untuk memahami lebih dalam tentang data. Berikut hasil analisis yang dilakukan:

- **Countplot pada Data Kategori**: Mayoritas pelanggan bekerja di sektor swasta, telah lulus kuliah, tidak memiliki status FrequentFlyer, dan tidak pernah melakukan perjalanan ke luar negeri.
- **Histplot pada Data Numerik**: Mayoritas pelanggan berusia sekitar 28 tahun, dengan pendapatan tahunan berkisar antara 300.000 hingga 1.800.000 rupee, dan memiliki 3 hingga 5 anggota keluarga.
- **Barplot Stacked pada Data Kategori dengan Hue TravelInsurance**: Pelanggan yang pernah melakukan perjalanan ke luar negeri cenderung lebih membeli asuransi perjalanan.
- **Pairplot dan Heatmap Correlation pada Data Numerik**: Terdapat korelasi positif yang rendah hingga sedang antara Age, AnnualIncome, dan FamilyMembers dengan keputusan pembelian asuransi perjalanan (TravelInsurance). Korelasi negatif yang rendah terlihat pada fitur ChronicDiseases.

## Data Preparation

Proses persiapan data sangat penting dalam pengembangan model prediktif. Berikut adalah beberapa langkah yang akan dilakukan dalam proses persiapan data:

- **Drop Data Duplikat**: Terdapat 738 baris data yang duplikat berdasarkan metode `.duplicated().sum()`, sehingga perlu dihapus. Menghapus duplikat data akan membantu menghindari bias dalam model dan memastikan integritas data yang baik. Duplikat data dapat mempengaruhi pembelajaran model dengan memberikan bobot yang tidak proporsional terhadap sampel tertentu.
- **Encoding Data Kategorikal**: Data kategorikal perlu diubah menjadi representasi numerik agar dapat digunakan dalam model pembelajaran mesin. Rencananya, akan dilakukan proses pengkodean kategori atau pemetaan (mapping) pada fitur Employment Type, GraduateOrNot, FrequentFlyer, dan EverTravelledAbroad. Pada fitur Employment Type, nilai "Government Sector" akan diubah menjadi 1 dan "Private Sector/Self Employed" akan diubah menjadi 0. Sedangkan pada fitur-fitur kategori lainnya, nilai "Yes" akan diubah menjadi 1 dan "No" akan diubah menjadi 0.
- **Pemisahan Data Train dan Test**: Data perlu dibagi menjadi set pelatihan dan pengujian dengan proporsi 0.8:0.2. Set pelatihan digunakan untuk melatih model, sedangkan set pengujian digunakan untuk mengevaluasi kinerja model. Ini penting untuk mengukur seberapa baik model akan berkinerja pada data yang belum pernah dilihat sebelumnya.
- **Oversampling dengan Metode SMOTE**: Ketidakseimbangan kelas dalam data train, yaitu data dengan label 0 berjumlah 616 sedangkan label 1 berjumlah 383, dapat mempengaruhi kinerja model. Metode SMOTE (Synthetic Minority Over-sampling Technique) digunakan untuk menyeimbangkan jumlah sampel antara kelas mayoritas dan minoritas dengan menciptakan sampel sintetis dari kelas minoritas.
- **Standarisasi dengan StandarScaler pada Fitur Age, FamilyMembers, dan AnnualIncome**: Standarisasi fitur numerik memastikan bahwa semua fitur memiliki skala yang serupa. Ini penting untuk algoritma yang sensitif terhadap skala, seperti regresi logistik atau SVM. Standarisasi juga membantu dalam konvergensi lebih cepat selama proses pembelajaran.


## Modeling
Pada tahap pemodelan, beberapa algoritma yang digunakan adalah sebagai berikut:

- Logistic Regression: Algoritma ini digunakan untuk tugas klasifikasi biner dan merupakan salah satu model yang paling sederhana dan mudah diinterpretasi. Kelebihannya termasuk interpretasi yang mudah, cocok untuk data yang memiliki fitur kategorikal, dan relatif cepat dalam pelatihan. Namun, kelemahannya adalah linearitas yang kuat, yang berarti mungkin tidak mampu menangani hubungan yang kompleks antara fitur dan target.
- RandomForestClassifier: Algoritma ini adalah ensambel dari pohon keputusan dan merupakan pilihan populer untuk klasifikasi. Kelebihannya termasuk kemampuan untuk menangani data dengan fitur yang tidak terstruktur atau tidak beraturan, serta toleran terhadap kelebihan fitting. Namun, kelemahannya adalah kompleksitas yang tinggi dan interpretasi yang sulit dibandingkan dengan model yang lebih sederhana seperti regresi logistik.
- GradientBoostingClassifier: Algoritma ini juga merupakan ensambel dari pohon keputusan, tetapi menggunakan pendekatan yang berbeda dengan RandomForest. Gradient boosting bekerja dengan cara menambahkan model yang berurutan, di mana setiap model berusaha untuk memperbaiki kesalahan yang dibuat oleh model sebelumnya. Kelebihannya termasuk kemampuan untuk menangani data yang tidak teratur dan kompleksitas model yang dapat diatur melalui hiperparameter. Namun, kelemahannya adalah cenderung overfit jika tidak diatur dengan benar.
- AdaBoostClassifier: Algoritma ini juga merupakan ensambel dari pohon keputusan, tetapi dengan pendekatan yang berbeda dari Gradient Boosting. Adaboost bekerja dengan cara memberikan bobot yang lebih besar pada sampel yang salah dikelasifikasi pada iterasi sebelumnya. Kelebihannya adalah kemampuan untuk menangani data yang tidak seimbang dan kemampuan untuk bekerja dengan baik dengan model yang sederhana. Namun, kelemahannya adalah rentan terhadap noise dan outlier dalam data.

Berdasarkan hasil pelatihan dan evaluasi, model GradientBoostingClassifier dipilih sebagai model terbaik dengan matriks evaluasi yang lebih tinggi dibandingkan model lainnya, terutama pada matriks ROC AUC dengan skor pelatihan 77% dan skor pengujian 75%.

## Evaluation
Matriks evaluasi yang digunakan meliputi Accuracy, Precision, Recall, F1 Score, dan ROC AUC. Dari semua metrik tersebut, ROC AUC dipilih sebagai matriks evaluasi utama karena kemampuannya untuk mengukur false positif dan false negatif, yang sangat penting dalam kasus klasifikasi yang tidak seimbang. 

ROC AUC adalah area di bawah kurva ROC (Receiver Operating Characteristic), yang menggambarkan hubungan antara laju true positive (sensitivitas) dan laju false positive (1-specificity). Sebuah model dengan ROC AUC yang tinggi menunjukkan bahwa model tersebut mampu memisahkan kelas positif dan negatif dengan baik, tanpa terpengaruh oleh ketidakseimbangan kelas.

Dengan memperhatikan matriks evaluasi ini, model klasifikasi terbaik dapat dipilih berdasarkan kinerjanya dalam memisahkan kelas positif dan negatif. Model dengan ROC AUC yang lebih tinggi akan dianggap lebih baik dalam memprediksi keputusan pembelian asuransi perjalanan, karena mampu mengidentifikasi pelanggan yang kemungkinan besar membeli asuransi perjalanan dengan lebih baik. Oleh karena itu, penggunaan ROC AUC sebagai matriks evaluasi utama akan membantu dalam pemilihan model yang paling sesuai dengan tujuan bisnis dan karakteristik data yang ada.

Berikut hasil matriks evaluasi dengan ROC AUC untuk setiap model:
<img source=assets/comparison-roc-auc.jpg>
