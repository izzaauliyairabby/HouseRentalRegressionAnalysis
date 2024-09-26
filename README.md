# Proyek Pertama House Rental Price Prediction - Izza Auliyai Rabby

Ini adalah proyek pertama dalam bidang predictive analytics yang dirancang untuk memenuhi submission di Dicoding. Proyek ini mengembangkan model machine learning yang mampu memprediksi harga sewa rumah dan apartemen di India.

## Domain Proyek

### Latar Belakang

Hunian seperti rumah atau apartemen merupakan kebutuhan dasar bagi manusia untuk berlindung dan menetap. Nilai dari hunian ini bergantung pada berbagai karakteristik, seperti lokasi, luas bangunan, jumlah kamar tidur, jumlah kamar mandi, kelengkapan perabot, serta fitur-fitur lainnya.

<br>

![brian-babb-XbwHrt87mQ0-unsplash](https://github.com/user-attachments/assets/8097bab4-ab5b-48e6-bb03-6dfcce73249c)


<br>

Harga setiap rumah ditentukan berdasarkan nilai yang dimiliki oleh properti tersebut. Namun, harga ini sering kali tidak pasti dan sulit diprediksi secara manual dengan akurat. Untuk mengurangi faktor ketidakpastian, perusahaan penyewaan perlu membangun sistem prediksi yang dapat menentukan harga sewa yang tepat berdasarkan karakteristik rumah tertentu.

Untuk mencapai tujuan tersebut, dilakukan penelitian untuk memprediksi harga sewa rumah menggunakan model machine learning. Diharapkan model ini mampu memberikan prediksi harga sewa yang sesuai dengan harga pasar. Prediksi ini nantinya akan menjadi acuan bagi perusahaan dalam menetapkan harga sewa rumah yang menguntungkan bagi bisnis mereka.

Referensi : House Price Prediction Using Multiple Linear Regression
Anirudh Kaushal, A. Shankar · Apr 25, 2021 
https://consensus.app/papers/house-price-prediction-using-multiple-linear-regression-kaushal/1e47e3d422145494907697f964e6da22/

## Business Understanding

Proyek ini dirancang untuk perusahaan dengan karakteristik bisnis sebagai berikut:

Perusahaan memiliki atau membeli rumah dan apartemen, kemudian menyewakannya kepada konsumen.
Perusahaan juga menyediakan layanan konsultasi mengenai harga sewa rumah dan apartemen bagi konsumen.

### Problem Statement

1. Faktor apa yang paling memengaruhi harga sewa rumah atau apartemen?
2. Bagaimana cara mempersiapkan data agar model dapat belajar dengan optimal?
3. Berapa perkiraan harga sewa rumah di pasaran berdasarkan berbagai karakteristik spesifik?

### Goals

1. Mengidentifikasi fitur yang memiliki pengaruh terbesar terhadap harga sewa rumah atau apartemen.
2. Melakukan pemrosesan data agar siap untuk dilatih oleh model.
3. Mengembangkan model machine learning yang mampu memprediksi harga sewa rumah dengan akurasi tinggi berdasarkan karakteristik tertentu.

### Solution Statement

1. Menganalisis data menggunakan univariate dan multivariate analysis, serta memvisualisasikan data untuk memahami hubungan antar fitur dan mendeteksi outlier.
2. Menyiapkan data agar siap digunakan dalam pembangunan model.
3. Melakukan hyperparameter tuning dengan grid search dan membangun model regresi untuk memprediksi nilai kontinu. Algoritma yang digunakan dalam proyek ini meliputi 4. K-Nearest Neighbour, Random Forest, dan AdaBoost.

## Data Understanding & Removing Outlier

Dataset yang digunakan dalam proyek ini merupakan data harga sewa rumah dengan berbagai karakteristik di India. Dataset ini dapat diunduh di [Kaggle : House Rent Prediction Dataset](https://www.kaggle.com/datasets/iamsouravbanerjee/house-rent-prediction-dataset).

Berikut informasi pada dataset :

Dataset tersedia dalam format CSV (Comma-Separated Values).
Dataset terdiri dari 4746 sampel dengan 12 fitur.
Terdapat 4 fitur bertipe int64 dan 8 fitur bertipe object.
Dataset tidak mengandung missing value.

### Variable - variable pada dataset

Posted On: Tanggal ketika data diunggah.
BHK: Jumlah kamar tidur, ruang tamu, dan dapur.
Rent: Biaya sewa rumah atau apartemen.
Size: Luas rumah atau apartemen dalam satuan kaki persegi (sqft).
Floor: Posisi lantai dan jumlah total lantai rumah atau apartemen.
Area Type: Jenis ukuran rumah, seperti Super Area, Carpet Area, atau Build Area.
Area Locality: Lokasi geografis rumah atau apartemen.
City: Kota tempat rumah atau apartemen berada.
Furnishing Status: Kondisi perabotan rumah atau apartemen, apakah Furnished, Semi-Furnished, atau Unfurnished.
Tenant Preferred: Jenis penyewa yang diutamakan oleh pemilik atau agen.
Bathroom: Jumlah kamar mandi yang tersedia.
Point of Contact: Kontak yang dapat dihubungi untuk informasi lebih lanjut tentang rumah atau apartemen.

Dari ke 12 fitur dapat dilihat bahwa fitur Point of Contract dan Posted On tidak mempengaruhi harga sewa rumah sehingga akan dihapus. Hal ini dikarenakan kedua fitur tersebut tidak diperlukan dalam membangun model prediksi harga sewa.

### Univariate Analysis

Univariate Analysis adalah menganalisis setiap fitur secara terpisah.

#### Analisis jumlah nilai unique pada setiap fitur kategorik

Fitur kategorik City, Furnishing Status, dan Tenant Preferred memiliki sebaran sample yang cukup merata.
<div><img src="https://user-images.githubusercontent.com/107544829/188319357-fc12fffa-b709-4584-8363-778bc678b328.png" width="220"/></div> <div><img src="https://user-images.githubusercontent.com/107544829/188319651-02ddb783-da3d-41ed-9b5f-9525aaaf9ed1.png" width="220"/></div> <div><img src="https://user-images.githubusercontent.com/107544829/188319750-1f080942-7826-4eaf-a021-8b9f938a861a.png" width="220"/></div><br />

Berikut adalah fitur dengan sample yang tidak merata :

+ Area Type
  <div><img src="https://user-images.githubusercontent.com/107544829/188318629-f474b626-a16a-4971-ab42-2c183d22b744.png" width="220"/></div>
  Hanya terdapat 2 sample Built Area pada fitur Area Type. Untuk menghindari high dimensional data, maka kedua sample ini akan dihapus.

+ Floor dan Area Locality
  <div><img src="https://user-images.githubusercontent.com/107544829/188319871-603b24b8-26b2-449b-b42e-59501a4803a7.png" width="220"/></div>
   <div><img src="https://user-images.githubusercontent.com/107544829/188319880-3226bd04-920e-4050-b5ab-38dec02fc524.png" width="220"/></div>
  Fitur Floor dan Area Locality memiliki banyak sekali nilai unique. Untuk menghindari high dimensional data, maka kedua fitur ini akan dihapus.

#### Analisis sebaran pada setiap fitur numerik

![1](https://github.com/user-attachments/assets/8de0426a-2ae2-45f5-ad3b-9543aa43e705)

Berikut analisis dari grafik di atas :

+ Sebagian besar rumah memiliki 1 sampai 3 BHK dan 1 sampai 3 kamar mandi.
+ Sebagian besar rumah memiliki luas di bawah 2000 sqft.
+ Rentang harga sewa cukup tinggi, yaitu dari 1200 hingga 3500000. Namun, rata-rata harga rumah hanya 35003. Distribusi harga yang kurang bagus seperti ini dapat berimplikasi pada model.

### Multivariate Analysis

Multivariate Analysis menunjukkan hubungan antara dua atau lebih fitur dalam data.

#### Analisis fitur numerik

Fitur Size dan BHK (Menghapus BHK Outlier)
Kedua fitur ini dianalisis karena tidak umum bagi rumah dengan 1 BHK memiliki luas 100 sqft. Oleh karena itu, ditentukan ambang batas sebesar 300 sqft per BHK. Data yang berada di bawah batas ini akan dihapus, yang mengakibatkan pengurangan jumlah sampel sebesar 548.

Fitur Size dan Rent (Menghapus Price per sqft Outlier)
Untuk mempermudah deteksi outlier, dibuat fitur baru bernama 'Price_per_sqft' dari kedua fitur ini untuk menganalisis harga sewa per luas sqft.

<div><img src="https://user-images.githubusercontent.com/107544829/188323140-6174b592-4c7b-4671-9acb-b49a621d2aba.png" width="220"/></div> Dari grafik tersebut, terlihat bahwa harga 571 per sqft sangat rendah, sementara harga 1.400.000 per sqft sangat tinggi. Oleh karena itu, outlier harga per sqft dihapus menggunakan rata-rata dan satu deviasi standar yang dikelompokkan berdasarkan kota. Proses ini menyebabkan pengurangan jumlah sampel sebesar 497.
Fitur Bathroom dan BHK (Menghapus Bathroom Outlier)
Kedua fitur ini dianalisis karena tidak lazim bagi rumah dengan 2 BHK memiliki 4 kamar mandi. Maka ditetapkan batasan bahwa jumlah kamar mandi tidak boleh melebihi jumlah BHK ditambah 2. Penghapusan ini menyebabkan berkurangnya jumlah sampel sebesar 3.
  
+ Melihat kolerasi antara semua fitur numerik
![2](https://github.com/user-attachments/assets/2400d97c-503a-4b88-8512-c48c17d7bf28)

 
Fitur BHK, Size, dan Bathroom menunjukkan korelasi yang tidak signifikan dengan fitur target (Rent). Hal ini mungkin disebabkan oleh keterbatasan data dalam penelitian ini. Di sisi lain, fitur BHK dan Bathroom memiliki korelasi yang signifikan dengan fitur Size, yang sesuai dengan harapan setelah proses penghapusan outlier yang telah dilakukan sebelumnya.

#### Analisis fitur kategorik

Analisis ini dilakukan untuk melihat kolerasi antara fitur kategorik dengan fitur target (Rent).

+ Fitur Area Type
  <div><img src="https://user-images.githubusercontent.com/107544829/188324455-9ae90db3-681a-4f14-bee0-0daaaec86490.png" width="500"/></div>
  Fitur Area Type memiliki pengaruh yang kecil terhadap rata-rata harga sewa.

+ Fitur City
  <div><img src="https://user-images.githubusercontent.com/107544829/188324564-b978b637-122b-403d-a760-eb0f7838bd95.png" width="500"/></div>
  Fitur City memiliki pengaruh cukup besar terhadap rata-rata harga sewa, terutama jika rumah berada di kota Mumbai. Hal ini dibuktikan dengan sebaran rumah yang mencapai harga tertinggi di kota Mumbai. Mumbai merupakan kota paling mahal di India untuk ditinggali, diikuti dengan Delhi.

  Referensi : [Ini Adalah Kota Termahal Untuk Hidup Di India](https://id.yourtripagent.com/these-are-most-expensive-cities-to-live-in-india-4734)

+ Fitur Furnishing Status
  <div><img src="https://user-images.githubusercontent.com/107544829/188324598-a765e404-4140-4518-91eb-fd298ba9d089.png" width="500"/></div>
  Fitur Furnishing Status memiliki pengaruh cukup besar terhadap rata-rata harga sewa. Merupakan hal biasa bila rumah yang memiliki perabotan lengkap akan diberi harga sewa lebih tinggi daripada rumah tanpa perabotan.

+ Fitur Tenant Preferred
  <div><img src="https://user-images.githubusercontent.com/107544829/188324642-0de4fe01-20c8-4560-981a-d6b0822d56ff.png" width="500"/></div>
  Fitur Tenant Preferred memiliki pengaruh yang lumayan terhadap rata-rata harga sewa. Dari grafik dapat terlihat bahwa rumah yang sangat disarankan untuk disewa oleh keluarga memiliki rata-rata harga sewa yang lebih mahal dibanding lainnya.

Persiapan Data
One Hot Encoding
One hot encoding adalah teknik yang digunakan untuk mengubah data kategorik menjadi data numerik, di mana setiap kategori diubah menjadi kolom baru dengan nilai 0 atau 1. Fitur yang akan diubah menjadi numerik dalam proyek ini adalah Area Type, City, Furnishing Status, dan Tenant Preferred.

Train Test Split
Proses train test split adalah membagi data menjadi dua bagian: data latih dan data uji. Data latih akan digunakan untuk membangun model, sementara data uji akan digunakan untuk menguji performa model. Dalam proyek ini, dataset sebesar 3696 dibagi menjadi 3511 untuk data latih dan 185 untuk data uji.

Normalisasi
Algoritma machine learning cenderung memiliki performa yang lebih baik dan beroperasi lebih cepat jika dimodelkan dengan data yang seragam dan memiliki skala yang relatif sama. Salah satu teknik normalisasi yang diterapkan dalam proyek ini adalah Standarisasi menggunakan sklearn.preprocessing.StandardScaler.

Pemodelan
Algoritma
Penelitian ini melakukan pemodelan menggunakan tiga algoritma, yaitu K-Nearest Neighbour, Random Forest, dan AdaBoost.

K-Nearest Neighbour
K-Nearest Neighbour bekerja dengan membandingkan jarak satu sampel ke sampel pelatihan lainnya dengan memilih sejumlah k tetangga terdekat. Proyek ini menggunakan sklearn.neighbors.KNeighborsRegressor dengan memasukkan X_train dan y_train untuk membangun model. Parameter yang digunakan dalam proyek ini adalah:

n_neighbors = Jumlah k tetangga terdekat.
Random Forest
Algoritma Random Forest adalah teknik dalam machine learning yang menggunakan metode ensemble. Teknik ini berfungsi dengan membangun banyak decision tree selama proses pelatihan. Proyek ini memanfaatkan sklearn.ensemble.RandomForestRegressor dengan memasukkan X_train dan y_train untuk membangun model. Parameter yang digunakan dalam proyek ini adalah:

n_estimators = Jumlah maksimum estimator di mana boosting dihentikan.
max_depth = Kedalaman maksimum setiap pohon.
random_state = Mengontrol seed acak yang diberikan pada setiap base_estimator selama setiap iterasi boosting.
AdaBoost
AdaBoost, atau Adaptive Boosting, adalah teknik dalam machine learning yang juga menggunakan metode ensemble. Algoritma yang paling umum digunakan dengan AdaBoost adalah pohon keputusan (decision trees) satu tingkat, yang dikenal sebagai Decision Stumps. Algoritma ini bertujuan untuk meningkatkan performa atau akurasi prediksi dengan menggabungkan beberapa model sederhana yang dianggap lemah (weak learners) secara berurutan, sehingga membentuk suatu model yang kuat (strong ensemble learner). Proyek ini menggunakan sklearn.ensemble.AdaBoostRegressor dengan memasukkan X_train dan y_train untuk membangun model. Parameter yang digunakan dalam proyek ini adalah:

n_estimators = Jumlah maksimum estimator di mana boosting dihentikan.
learning_rate = Learning rate yang memperkuat kontribusi setiap regressor.
random_state = Mengontrol seed acak yang diberikan pada setiap base_estimator selama setiap iterasi boosting.

+ Hyperparameter Tuning (Grid Search)
  Hyperparameter tuning adalah cara untuk mendapatkan parameter terbaik dari algoritma dalam membangun model. Salah satu teknik dalam hyperparameter tuning yang digunakan dalam proyek ini adalah grid search. Berikut adalah hasil dari Grid Search pada proyek ini :

model	best_score	best_params
0	knn	0.460230	{'n_neighbors': 7}
1	boosting	0.856539	{'learning_rate': 0.1, 'n_estimators': 100, 'r...
2	random_forest	0.893655	{'max_depth': 8, 'n_estimators': 25, 'random_s...


## Evaluation

Metrik evaluasi yang digunakan pada proyek ini adalah akurasi dan mean squared error (MSE). Akurasi menentukan tingkat kemiripan antara hasil prediksi dengan nilai yang sebenarnya (y_test). Mean squared error (MSE) mengukur error dalam model statistik dengan cara menghitung rata-rata error dari kuadrat hasil aktual dikurang hasil prediksi. Berikut formulan MSE :

![4](https://github.com/user-attachments/assets/7e06fc53-9b2d-431b-9732-53a1af561d7c)
![5](https://github.com/user-attachments/assets/c5cd3ec6-4ac1-4684-8f68-9c93613705ba)


Berikut hasil evaluasi pada proyek ini :

+ Akurasi
  | model    | accuracy |
  |----------|----------|
  | knn      | 0.726986 |
  | boosting | 0.932057 |
  | rf       | 0.898556 |			


+ Mean Absolute Error (MAE)


Dari hasil evaluasi dapat dilihat bahwa model dengan algoritma Random Forest memiliki akurasi lebih tinggi tinggi dan tingkat error lebih kecil dibandingkan algoritma lainnya dalam proyek ini.
![6](https://github.com/user-attachments/assets/b3294956-6ab4-48a5-b193-866e1523bffa)
