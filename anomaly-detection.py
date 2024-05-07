//# Import modules

import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
pd.set_option("display.float", "{:.2f}".format)
"%matplotlib inline"
sns.set_style("whitegrid")
from sklearn.ensemble import IsolationForest
from datetime import datetime

//# Read from csv file
data_1_month = pd.read_csv('./data/1_aylik.csv', sep=';')

//# Sutunlari cikarma
data_1_month = data_1_month.drop(columns=['EUR/MWh','TL/MWh','Tarih','Saat'])

data = pd.DataFrame(data_1_month)
//# Fiyat sutununu turunu degistirme
data['USD/MWh'] = data['USD/MWh'].astype(float)
Fiyat = data['USD/MWh']
//# Tarih/Saat sutunun turunu degistirme
data['Tarih/Saat'] = pd.to_datetime(data['Tarih/Saat'])
Zaman = data['Tarih/Saat']
# data['Tarih/Saat_formatted'] = data['Tarih/Saat'].dt.strftime('%mm.%dd/%Y %H:%M')


# print(data_1_month)
# print(data.dtypes)
# print(Fiyat)
/*
# # MATPLOTLIB ile zaman grafigi cizimi
# plt.plot(Zaman,Fiyat)  # Zaman eksenine göre fiyatları çiz
# plt.xlabel("Zaman")  # X eksenini etiketle
# plt.ylabel("Fiyat")  # Y eksenini etiketle
# plt.title("Fiyat Zaman Grafiği")  # Grafiğe başlık ekle
# plt.grid(True)  # Grafiğe ızgara ekle
# plt.show()  # Grafiği göster
*/
//# Model olustur
model = IsolationForest(n_estimators=100, contamination=0.5)

//# Modeli egit
model.fit(Fiyat.values.reshape(-1,1))

//# Tahminleyiciyi tanimla
anomaly_predictions = model.predict(Fiyat.values.reshape(-1, 1))

//# Grafigi olustur
plt.plot(Zaman, Fiyat, label="Normal Veriler")
plt.scatter(Zaman[anomaly_predictions == -1], Fiyat[anomaly_predictions == -1], c="red", label="Anormal Veriler")
plt.legend()
plt.show()


//# Anomali skorlarını 'fiyat_sutunu' verileri için hesaplayın
anomaly_scores = model.decision_function(Fiyat.values.reshape(-1, 1))
//# Eşik değeri belirleyin (örneğin, 0.5)
esik_degeri = 0.5

//# Anormal veri noktalarını belirleyin
anormal_veriler = Fiyat[anomaly_scores < esik_degeri]

//# Anormal veri noktalarını ve anomali skorlarını inceleyin
print("Anormal Veriler:")
print(anormal_veriler)

print("Anomali Skorları:")
print(anomaly_scores[anomaly_scores < esik_degeri])
/*
# Anomali Tahmini:
# Modelin her bir veri noktası için verdiği "anormal" veya "normal" olma kararıdır.
# İkili bir çıktıdır, ya -1 (anormal) ya da 1 (normal) olarak temsil edilir.
# Model, verilerin genel yapısına uymayan veri noktalarını anormal olarak sınıflandırır.

# Anomali Skoru:
# Modelin, bir veri noktasının anormal olma olasılığını tahmin etmek için kullandığı sayısal değerdir.
# Skoru ne kadar düşükse, veri noktasının anormal olma olasılığı o kadar yüksektir.
# Model, karar ağaçlarının her katmanında bir izolasyon puanı atar ve bu puanlar nihai anomali skoru hesaplamak için birleştirilir.
# Farkları:
# Çıktı Türü: Anomali tahmini kategorik iken (anormal veya normal), anomali skoru sayısal bir değerdir.
*/










# Tarih ve Saat sutunlarini birlestirme

# ## Dataframe concat ile cevirip birlestirme CALISMADI
# Tarih = pd.DataFrame(data_1_month['Tarih'], columns=['Tarih'])
# Saat = pd.DataFrame(data_1_month['Saat'], columns=['Saat'])
# DateTime = pd.concat([Tarih,Saat],axis=1)
# data_1_month['DateTime'] = DateTime
# print(data_1_month)


# ## stringe cevirip birlestirme CALISMADI
# Tarih_str = data_1_month['Tarih'].astype(str)
# Saat_str = data_1_month['Saat'].astype(str)
# birlesik = Tarih_str + Saat_str
# data_1_month['birlesik'] = birlesik
# print(data_1_month)
# print(data_1_month.columns)


# Tarih = pd.to_datetime(data_1_month['Tarih'])
# Saat = pd.to_datetime(data_1_month['Saat'])
# combine_column = Tarih + Saat
# print(combine_column)



