import os
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

# ==========================================
# 1. ADIM: VERİ YÜKLEME VE İŞLEME (Akıllı Bölme)
# ==========================================
print("Sistem-başlat-ayarlar: Veriler hazırlanıyor...")

baslik_ayarlari = [
    ("#", "Bolum"),
    ("##", "Hastalik")
]
metin_bolucu = MarkdownHeaderTextSplitter(headers_to_split_on=baslik_ayarlari)

# Markdown formatındaki dosyamızı okuyoruz
with open("hastaliklar.txt", "r", encoding="utf-8") as dosya:
    okunan_metin = dosya.read()

ayrilmis_metinler = metin_bolucu.split_text(okunan_metin)
print(f"{len(ayrilmis_metinler)} adet hastalık protokolü başarıyla ayrıldı.")


# ==========================================
# 2. ADIM: YEREL HAFIZA VE VEKTÖR VERİTABANI
# ==========================================
print("Sistem-başlat-ayarlar: Sayısal hafıza (Vektör DB) kontrol ediliyor...")

# Ollama ile Llama3 modelini kullanarak Embedding (Vektörleştirme) motorunu kuruyoruz
sayisal_donusturucu = OllamaEmbeddings(
    model="llama3",
    base_url="http://host.docker.internal:11434" # Ollama'nın çalıştığı adres
)

kayit_klasoru = "./chroma_db" # Veritabanının kaydedileceği klasör

# Eğer daha önce veritabanı oluşturulduysa onu yükle, yoksa sıfırdan oluştur
if os.path.exists(kayit_klasoru):
    print("Mevcut veritabanı bulundu, hafıza yükleniyor...")
    vektor_veritabani = Chroma(
        persist_directory=kayit_klasoru, 
        embedding_function=sayisal_donusturucu
    )
else:
    print("İlk çalıştırma tespit edildi. Kelimeler sayılara çevriliyor ve veritabanı oluşturuluyor...")
    vektor_veritabani = Chroma.from_documents(
        documents=ayrilmis_metinler, 
        embedding=sayisal_donusturucu, 
        persist_directory=kayit_klasoru
    )

# Veri getirme mekanizması (Kütüphaneci)
bilgi_getirici = vektor_veritabani.as_retriever(search_kwargs={"k": 2}) 
# "k=2" demek: Hastanın şikayetine en çok benzeyen 2 hastalığı getir demektir.

print("\n--- 1. VE 2. ADIM BAŞARIYLA TAMAMLANDI ---")
print("Sistem şu an hastayı dinlemeye ve veritabanında arama yapmaya hazır.")