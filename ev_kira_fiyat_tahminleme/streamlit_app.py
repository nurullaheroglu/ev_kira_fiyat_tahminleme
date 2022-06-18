import time
import streamlit as st
import joblib
import pandas as pd
from helper import *

model = joblib.load("ev_kira_fiyat_tahminleme/final_model.pkl")
df1 = pd.read_csv("ev_kira_fiyat_tahminleme/house_rent_prices_dataset.csv", index_col="Unnamed: 0").reset_index().drop("index", axis=1)

anadolu_list = set([ilce for i, ilce in enumerate(df1["ilce"]) if df1["yaka"][i] == "Anadolu"])
avrupa_list = set([ilce for i, ilce in enumerate(df1["ilce"]) if df1["yaka"][i] == "Avrupa"])


st.markdown("<h1 style='text-align: center; color: ##00a6f9;'>Ev Kira Fiyatı Tahminleme</h1>", unsafe_allow_html=True)
st.markdown("<h5 style='text-align: center; color: #1B9E91;'>Lightgbm makine öğrenmesi yöntemini kullanarak geliştirdiğimiz bu modelde; İstanbul içindeki evlerin fiyatlarını tahmin ediyoruz.</h5>", unsafe_allow_html=True)

option_yaka = st.sidebar.selectbox('Ev hangi yakada ?', ('Anadolu', 'Avrupa'))

if option_yaka == "Anadolu":
    option_ilce = st.sidebar.selectbox(
        'İlce seçiniz',
        anadolu_list)
else:
    option_ilce = st.sidebar.selectbox(
        'İlce seçiniz',
        avrupa_list)

option_m2_brut = st.sidebar.number_input('Metrekare brüt bilgisini giriniz.', 0, 500, step=1)

option_bina_yasi = st.sidebar.selectbox('Bina yaş aralığını seçiniz.',
                                        ('0-5 arası','5-10 arası','11-15 arası', '16-20 arası','21-25 arası','26-30 arası','31 ve üzeri'))

option_kat_sayisi = st.sidebar.slider('Binanın kat sayısını seçiniz.', 1, 40)

option_bulundugu_kat = st.sidebar.slider('Bulunduğu katı seçiniz.', -1, option_kat_sayisi)

option_isitma = st.sidebar.selectbox(
     'Isıtma türünü seçiniz.',
     ('Yerden Isıtma','Merkezi (Pay Ölçer)','Doğalgaz (Kombi)','Merkezi','Doğalgaz Sobası','Yok','Klima','Kat Kaloriferi','Fancoil Ünitesi','Soba','VRV','Elektrikli Radyatör'))

option_banyo_sayisi = st.sidebar.number_input('Banyo sayısını seçiniz.', 1, 8)

option_balkon = st.sidebar.selectbox('Balkon var mı ?', ('Yok', 'Var'))

option_esyali = st.sidebar.selectbox('Ev eşyalı mı ?', ('Evet', "Hayır"))

option_site_icerisinde = st.sidebar.selectbox('Site içerisinde mi ?', ('Evet', "Hayır"))

option_kimden = st.sidebar.selectbox('Kimden kiralık olsun ?', ('Emlak Ofisinden', 'Sahibinden', 'İnşaat Firmasından'))

option_oda = st.sidebar.number_input('Kaç odası var ?', 1, 10)

option_salon = st.sidebar.number_input('Kaç tane salonu var ?', 0, 5)

option_merkeze_yakin = st.sidebar.selectbox('Merkeze yakın mı ?', ('Hayır', 'Evet'))

option_aidat_aralik = st.sidebar.selectbox('Verebileceğiniz aidat aralığını seçiniz', ('750 - 1000', '300 - 500', '500 - 750', '0 - 150', '150 - 300'))

option_luks_mu = st.sidebar.selectbox('Lüks mü ?', ('Hayır', 'Evet'))


new_user = {"ilce": option_ilce,
            "m2_brut": option_m2_brut,
            "bina_yasi": option_bina_yasi,
            "bulundugu_kat": option_bulundugu_kat,
            "kat_sayisi": option_kat_sayisi,
            "isitma": option_isitma,
            "banyo_sayisi": option_banyo_sayisi,
            "balkon": option_balkon,
            "esyali": option_esyali,
            "site_icerisinde": option_site_icerisinde,
            "kimden": option_kimden,
            "yaka": option_yaka,
            "oda": option_oda,
            "salon": option_salon,
            "merkeze_yakin": option_merkeze_yakin,
            "aidat_aralik": option_aidat_aralik,
            "luks_mu": option_luks_mu}

new_user = pd.Series(new_user)
df1 = df1.append(pd.Series(new_user, index=df1.columns[:len(new_user)]), ignore_index=True)
df2 = new_features(df1)
df3 = model_prep(df2)
X = df3.drop("fiyat", axis=1)

user = X.iloc[[-1]]

col1, col2, col3 = st.columns(3)
my_bar = st.progress(0)

#foto ekleme
from PIL import Image
image = Image.open('ev_kira_fiyat_tahminleme/photo.jpg')
st.image(image, use_column_width="always")

with col2:
    if st.button("Kira Fiyatını Hesapla"):
        for percent_complete in range(100):
            time.sleep(0.05)
            my_bar.progress(percent_complete + 1)
        I1 = round(model.predict(user)[0] / 100 * 90, 0)
        I4 = round(model.predict(user)[0] / 100 * 110, 0)
        with col2:
            st.success(f"Önerdiğimiz Fiyat Aralığı\n"
                       f"Minimum: {I1} TL\n"
                       f"Maksimum: {I4} TL")


        st.balloons()




