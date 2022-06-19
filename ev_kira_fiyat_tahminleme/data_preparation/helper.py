import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer


def new_features(df):
    df["kat_bina_orani"] = df["bulundugu_kat"].astype("int64") / df["kat_sayisi"].astype("int64")
    df = df[df["kat_bina_orani"] < 2].reset_index().drop("index", axis=1)
    df["yukseklik"] = pd.qcut(df["kat_bina_orani"], 4, labels=["alçak", "ort-yük", "yüksek", "çok-yük"])
    df["arakat_mi"] = ["Evet" if ((kat < 1) & (kat > 0)) else "Hayır" for kat in df["kat_bina_orani"]]
    df = df.drop("kat_bina_orani", axis=1)
    return df


def model_prep(df):
    df_model = df.copy()

    def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
        dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
        return dataframe

    num_cols = []
    cat_cols = []
    for i in df_model.columns:
        if df_model[i].dtype not in ["float64", "int64"]:
            cat_cols.append(i)
        else:
            num_cols.append(i)

    df_model = one_hot_encoder(df_model, cat_cols, drop_first=True)
    print("Process complete.")
    return df_model


def main_df_generate(df):
    print("Generating features...")
    esy = df.copy()
    esy["esyali"] = esy["esyali"].fillna("Belirtilmemiş")
    # eşyalı mı
    for i, a in enumerate(esy["aciklama"]):
        if (esy["esyali"][i] == "Belirtilmemiş"):
            if (("eşya" in a) | ("esya" in a)):
                df["esyali"][esy.index[i]] = "Evet"
            else:
                df["esyali"][esy.index[i]] = "Hayır"

    # oda ayrımı
    df = df[df["oda_sayisi"] != "10 Üzeri"]

    for i in df["oda_sayisi"].values:
        if "Stüdyo" in i:
            df = df.replace(i, np.nan, regex=False)

    df["oda_sayisi"] = df["oda_sayisi"].fillna("1+0")
    sep_oda = [str(i).split("+") for i in df["oda_sayisi"]]
    df["oda"] = [float(i[0]) for i in sep_oda]
    df["salon"] = [float(i[1]) for i in sep_oda]
    df.drop("oda_sayisi", axis=1, inplace=True)

    # kat ile bina yüksekliği oranı - arakat mı
    from sklearn.preprocessing import MinMaxScaler
    df = df[(df["bulundugu_kat"] != "Villa Tipi") & (df["bulundugu_kat"] != "Müstakil")]
    df["bulundugu_kat"] = [
        1 if i in "Yüksek Giriş" else "max" if i in ["Çatı Katı"] else 0 if i in ["Bahçe Katı", "Giriş Katı",
                                                                                  "Giriş Altı Kot 1",
                                                                                  "Zemin Kat"] else -1 if i in [
            "Bodrum Kat", "Giriş Altı Kot 2", "Giriş Altı Kot 3", "Giriş Altı Kot 4"] else i for i in
        df["bulundugu_kat"]]
    df["kat_sayisi"] = ["31" if i == "30 ve üzeri" else i for i in df["kat_sayisi"]]
    df.loc[df["bulundugu_kat"] == "30 ve üzeri", "bulundugu_kat"] = df["kat_sayisi"]
    df.loc[df["bulundugu_kat"] == "max", "bulundugu_kat"] = df["kat_sayisi"]

    # bina yaşı
    df["bina_yasi"] = ["0-5 arası" if i in ["0", "4", "3", "2", "1"] else i for i in df["bina_yasi"]]

    # yeni düzenlemeler
    df = df.drop_duplicates()
    df = df.reset_index()
    df = df.drop("index", axis=1)

    # mahalleyi düşürmek
    df.drop("mahalle", axis=1, inplace=True)

    # banyo
    df["banyo_sayisi"] = [sayi if sayi not in ["Yok", "6 Üzeri", "5"] else
                          np.nan for sayi in df["banyo_sayisi"]]
    df["banyo_sayisi"] = df["banyo_sayisi"].fillna("1")

    # merkeze yakınlık
    df["merkeze_yakin"] = ["Evet" if (("merkez" in aciklama) | ("meydan" in aciklama)) else "Hayır" for aciklama in
                           df["aciklama"]]

    df["bulundugu_kat"] = df["bulundugu_kat"].astype("int64")
    df["kat_sayisi"] = df["kat_sayisi"].astype("int64")
    df["banyo_sayisi"] = df["banyo_sayisi"].astype("int64")
    

    def knn_impute_aidat(df):
        dff = df.drop("aciklama", axis=1)
        dff = pd.get_dummies(dff, drop_first=True)

        # değişkenlerin standartlatırılması
        scaler = MinMaxScaler()
        dff = pd.DataFrame(scaler.fit_transform(dff), columns=dff.columns)

        # knn'in uygulanması.
        imputer = KNNImputer(n_neighbors=5)
        dff = pd.DataFrame(imputer.fit_transform(dff), columns=dff.columns)
        dff = pd.DataFrame(scaler.inverse_transform(dff), columns=dff.columns)
        df["aidat_imputed_knn"] = dff[["aidat"]]
        df.drop("aidat", axis=1, inplace=True)
        df["aidat_aralik"] = pd.cut(df["aidat_imputed_knn"], [0, 150, 300, 500, 750, 1000],
                                    labels=["0 - 150", "150 - 300", "300 - 500", "500 - 750", "750 - 1000"])
        df.drop("aidat_imputed_knn", axis=1, inplace=True)
        print("Process complete.")
        return df

    df = knn_impute_aidat(df)

    try:
        list_ = [aciklama for aciklama in df["aciklama"] if (("ev arkadaşı" in aciklama) |
                                                             ("arkadaş" in aciklama) |
                                                             ("ev arkadasi" in aciklama) |
                                                             ("arkadas" in aciklama) |
                                                             ("bayan" in aciklama) |
                                                             ("bay" in aciklama) |
                                                             ("kadın" in aciklama) |
                                                             ("kadin" in aciklama) |
                                                             ("erkek" in aciklama))]
        df["aciklama"] = [np.nan if aciklama in list_ else aciklama for aciklama in df["aciklama"]]
        df = df.dropna()
        df = df.reset_index()
        df.drop("index", axis=1, inplace=True)
    except:
        pass
    # lüks mü
    df["luks_mu"] = ["Evet" if (("lüks" in aciklama) | ("lüx" in aciklama) |
                                ("luks" in aciklama) | ("lux" in aciklama) | ("ultra" in aciklama))
                     else "Hayır" for aciklama in
                     df["aciklama"]]
    df.drop("aciklama", axis=1, inplace=True)
    df = new_features(df)
    df = model_prep(df)
    X = df.drop("fiyat", axis=1)
    y = df["fiyat"]
    return X, y, df
