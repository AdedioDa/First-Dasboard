import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Judul Dashboard
st.title("Dashboard Analisis Data Bike Sharing")
st.subheader("Proyek Analisis Data: Bike Sharing Dataset")
st.write("Nama: **Adedio Daniel Situmeang**")
st.write("Email: **adediodaniel9@gmail.com**")
st.write("ID Dicoding: **Adedio Daniel S**")

# Memuat data
data = pd.read_csv('day.csv')
data['dteday'] = pd.to_datetime(data['dteday'])
data['month'] = data['dteday'].dt.month
data['day_type'] = data['weekday'].apply(lambda x: 'Weekend' if x in [5, 6] else 'Weekday')

# Sidebar untuk memilih jenis analisis
st.sidebar.title("Pilih Analisis")
analysis_type = st.sidebar.selectbox("Pilih jenis analisis:", 
                                     ["Tren Penggunaan Sepeda", "Korelasi Cuaca dan Penyewaan", "RFM Segmentation"])

# Analisis 1: Tren Penggunaan Sepeda oleh Penyewa Terdaftar dan Kasual
if analysis_type == "Tren Penggunaan Sepeda":
    st.header("Tren Penggunaan Sepeda oleh Penyewa Terdaftar dan Kasual")
    
    data_grouped = data.groupby(['yr', 'month'])[['casual', 'registered']].sum()
    fig, ax = plt.subplots(figsize=(10, 5))
    data_grouped.plot(kind='bar', ax=ax)
    plt.title('Penggunaan Sepeda Kasual dan Terdaftar Berdasarkan Bulan')
    plt.xlabel('Tahun, Bulan')
    plt.ylabel('Jumlah Pengguna')
    st.pyplot(fig)

    # Insight
    st.write("### Insight:")
    st.write("""
    - Pengguna kasual lebih aktif pada bulan tertentu seperti musim panas, sedangkan penyewa terdaftar lebih stabil sepanjang tahun.
    - Puncak peminjaman sepeda terjadi pada pertengahan tahun.
    """)

# Analisis 2: Korelasi Cuaca dan Jumlah Penyewaan Sepeda
elif analysis_type == "Korelasi Cuaca dan Penyewaan":
    st.header("Korelasi Antara Cuaca, Suhu, dan Jumlah Penyewaan Sepeda")
    
    correlation = data[['weathersit', 'temp', 'cnt']].corr()
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(correlation, annot=True, cmap='coolwarm', ax=ax)
    plt.title('Korelasi antara Cuaca, Suhu, dan Jumlah Peminjaman')
    st.pyplot(fig)

    st.write("### Insight:")
    st.write("""
    - Faktor suhu (temp) memiliki hubungan positif dengan jumlah peminjaman.
    - Cuaca buruk (weathersit) menunjukkan hubungan negatif dengan peminjaman sepeda.
    """)

    # Visualisasi Rata-rata Peminjaman Berdasarkan Kondisi Cuaca
    st.subheader("Rata-rata Jumlah Peminjaman Berdasarkan Kondisi Cuaca")
    weather_grouped = data.groupby(['weathersit'])[['cnt']].mean()
    fig, ax = plt.subplots(figsize=(8, 5))
    weather_grouped.plot(kind='bar', color='orange', ax=ax)
    plt.title('Rata-rata Jumlah Peminjaman Berdasarkan Kondisi Cuaca')
    plt.xlabel('Kondisi Cuaca (1: Clear, 2: Mist, 3: Light Rain)')
    plt.ylabel('Rata-rata Jumlah Peminjaman')
    st.pyplot(fig)

# Analisis 3: Segmentasi Pelanggan Menggunakan RFM
elif analysis_type == "RFM Segmentation":
    st.header("Segmentasi Pelanggan Menggunakan RFM")
    
    # Menghitung Recency, Frequency, dan Monetary
    data['last_day'] = data.groupby('registered')['dteday'].transform('max')
    frequency = data.groupby('registered')['cnt'].count().reset_index()
    frequency.columns = ['registered', 'frequency']
    monetary = data.groupby('registered')['cnt'].sum().reset_index()
    monetary.columns = ['registered', 'monetary']

    rfm = pd.merge(frequency, monetary, on='registered')
    recent_date = data['dteday'].max()
    rfm['recency'] = (recent_date - data['last_day']).dt.days

    # Skor RFM
    rfm['R_score'] = pd.qcut(rfm['recency'], 5, labels=[5, 4, 3, 2, 1], duplicates='drop')
    frequency_bins = pd.cut(rfm['frequency'], bins=5, labels=[1, 2, 3, 4, 5], duplicates='drop')
    rfm['F_score'] = frequency_bins
    monetary_bins = pd.cut(rfm['monetary'], bins=5, labels=[1, 2, 3, 4, 5], duplicates='drop')
    rfm['M_score'] = monetary_bins
    rfm['RFM_Score'] = rfm['R_score'].astype(str) + rfm['F_score'].astype(str) + rfm['M_score'].astype(str)

    # Segmentasi berdasarkan RFM Score
    def segment_rfm(df):
        if df['RFM_Score'] == '555':
            return 'Best Customers'
        elif df['R_score'] == 5:
            return 'Loyal Customers'
        elif df['R_score'] <= 2:
            return 'Lost Customers'
        elif df['F_score'] >= 4:
            return 'Frequent Customers'
        else:
            return 'Others'
    
    rfm['Segment'] = rfm.apply(segment_rfm, axis=1)

    # Visualisasi Hasil RFM
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.countplot(data=rfm, x='Segment', order=rfm['Segment'].value_counts().index, ax=ax)
    plt.title('Distribusi Segmen Pelanggan Berdasarkan Skor RFM')
    plt.xlabel('Segmen Pelanggan')
    plt.ylabel('Jumlah Pengguna')
    st.pyplot(fig)

    # Menampilkan Data RFM
    st.write("### Data RFM Segmentation")
    st.dataframe(rfm.head())
