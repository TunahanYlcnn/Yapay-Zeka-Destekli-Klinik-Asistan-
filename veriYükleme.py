# Dosya İsmi: veri_yukleme.py

from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings # YENİ EKLENDİ
from langchain_core.documents import Document
import os
import shutil # Sistem-Klasör-Ayarları için eklendi

print("Sistem-Başlatılıyor: Veritabanı oluşturma işlemi başlıyor...")

# 1. ESKİ VERİTABANINI TEMİZLEME
# Eski ve yanlış vektörlerin tamamen silindiğinden emin oluyoruz.
klasor_yolu = "./chroma_db"
if os.path.exists(klasor_yolu):
    print(f"Eski ve bozuk {klasor_yolu} klasörü siliniyor...")
    shutil.rmtree(klasor_yolu)
    print("Temizlik tamamlandı.")

# 2. DOĞRU EMBEDDİNG MODELİNİ AYARLAMA (Türkçe Destekli)
# DİKKAT: Burada sadece metinleri sayılara çevirme (vektörleştirme) işinde uzman olan modeli kullanıyoruz.
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

# 3. YÜKLENECEK VERİLER (Klinik Asistan İçin)
# txt dosyasından okuma işlemi yapılıyor ve veriler ayrıştırılıyor.
dokumanlar = []

print("Metin dosyası okunuyor ve veriler ayrıştırılıyor...")
try:
    with open("hastaliklar.txt", "r", encoding="utf-8") as dosya:
        icerik = dosya.read()
        
        # İçeriği "## Hastalık:" kelimesine göre parçalara bölüyoruz. 
        # Böylece her hastalık kendi içinde bütün kalır, veritabanı kafası karışmaz.
        hastalik_bolumleri = icerik.split("## Hastalık:")
        
        for bolum in hastalik_bolumleri:
            temiz_bolum = bolum.strip()
            # Başlık kısmı veya boşlukları atlayıp, sadece hastalık verisi içeren kısımları alıyoruz.
            if "Belirtiler:" in temiz_bolum:
                # Böldüğümüzde silinen "Hastalık:" kelimesini başa tekrar ekliyoruz
                tam_metin = f"Hastalık: {temiz_bolum}"
                dokumanlar.append(Document(page_content=tam_metin))
                
except FileNotFoundError:
    print("\nSistem-Hatası: 'hastaliklar.txt' dosyası bulunamadı!")
    print("Lütfen dosyanın veri_yukleme.py ile aynı klasörde olduğundan emin olun.")
    exit()

# 4. VERİTABANINI OLUŞTURMA VE KAYDETME
print("\nVeriler matematiksel vektörlere dönüştürülüyor (Bu işlem bilgisayarın hızına göre birkaç saniye sürebilir)...")

# from_documents komutu verileri alır, embeddings ile sayılara çevirir ve persist_directory içine kaydeder.
vectorstore = Chroma.from_documents(
    documents=dokumanlar,
    embedding=embeddings,
    persist_directory=klasor_yolu
)

# 5. SONUÇ KONTROLÜ
kayit_sayisi = vectorstore._collection.count()
print(f"\nSistem-Başarılı: Veritabanı txt dosyasından sıfırdan oluşturuldu!")
print(f"Toplam {kayit_sayisi} adet hastalık sisteme kaydedildi.")