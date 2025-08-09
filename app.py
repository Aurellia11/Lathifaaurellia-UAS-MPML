import streamlit as st
import pandas as pd
import joblib
import os
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Konfigurasi tampilan
st.set_page_config(page_title="Prediksi Profitabilitas Menu", layout="wide")

# Judul Aplikasi
st.title("🍽️ Prediksi Profitabilitas Menu Restoran")
st.markdown("""
Aplikasi ini memprediksi apakah suatu menu restoran akan profitable berdasarkan kategori dan harga.
""")

# Sidebar untuk input user
with st.sidebar:
    st.header("Parameter Input")
    menu_category = st.selectbox(
        "Kategori Menu",
        ["Makanan", "Minuman", "Dessert"],
        index=0
    )
    
    price = st.slider(
        "Harga Menu (USD)",
        min_value=1.0,
        max_value=100.0,
        value=15.0,
        step=0.5
    )
    
    st.markdown("---")
    st.info("Pastikan semua parameter sudah diisi sebelum memprediksi")

# Fungsi untuk load model
@st.cache_resource
def load_model():
    try:
        model = joblib.load('models/best_model.pkl')
        return model
    except Exception as e:
        st.error(f"Gagal memuat model: {str(e)}")
        return None

# Fungsi preprocessing
@st.cache_resource
def load_model():
    try:
        # Load model yang sudah termasuk preprocessor fitted
        model = joblib.load('models/best_model.pkl')
        st.success("Model dan preprocessor berhasil dimuat!")
        return model
    except Exception as e:
        st.error(f"Gagal memuat model: {str(e)}")
        return None

# Main Program
def main():
    model = load_model()
    
    if model is not None:
        # Buat DataFrame dari input
        input_data = pd.DataFrame({
            'MenuCategory': [menu_category],
            'Price': [price]
        })
        
        # Preprocessing
        if st.button("☑ Prediksi Sekarang", type="primary"):
        prediction = model.predict(input_data)  # Model akan handle preprocessing otomatis
        
        # Prediksi
        if st.button("🚀 Prediksi Sekarang", type="primary"):
            try:
                prediction = model.predict(processed_data)
                proba = model.predict_proba(processed_data)
                
                # Tampilkan hasil
                col1, col2 = st.columns(2)
                with col1:
                    if prediction[0] == 1:
                        st.success("## ✅ Profitabel")
                    else:
                        st.error("## ❌ Tidak Profitabel")
                
                with col2:
                    st.metric(
                        label="Probabilitas",
                        value=f"{max(proba[0])*100:.1f}%"
                    )
                
                # Interpretasi
                st.markdown("---")
                st.subheader("📊 Interpretasi")
                if prediction[0] == 1:
                    st.write("""
                    Menu ini memiliki karakteristik yang menguntungkan berdasarkan model kami.
                    Pertimbangkan untuk mempertahankan harga dan kategori ini.
                    """)
                else:
                    st.write("""
                    Menu ini mungkin memerlukan penyesuaian harga atau perubahan kategori
                    untuk meningkatkan profitabilitas.
                    """)
                    
            except Exception as e:
                st.error(f"Terjadi error saat prediksi: {str(e)}")

    # Debug info (opsional)
    with st.expander("ℹ️ Informasi Teknis"):
        st.write("""
        **Model yang digunakan:** Random Forest Classifier  
        **Akurasi model:** 85% (pada data testing)  
        **Fitur penting:** Harga dan Kategori Menu  
        """)
        if st.button("Tampilkan Struktur Folder"):
            st.code(f"""
            {os.listdir()}
            {os.listdir('models') if os.path.exists('models') else 'Folder models tidak ditemukan'}
            """)

if __name__ == "__main__":
    main()
