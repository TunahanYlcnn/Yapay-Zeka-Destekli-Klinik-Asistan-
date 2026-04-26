# Dosya İsmi: veriYükleme.py
import os
import shutil
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import MarkdownHeaderTextSplitter

print("--- [ADIM 1] Veritabanı Oluşturma Başlatılıyor ---")

# 1. ESKİ VERİTABANINI TEMİZLEME
klasor_yolu = "./chroma_db"
if os.path.exists(klasor_yolu):
    print(f"Eski veritabanı ({klasor_yolu}) siliniyor...")
    shutil.rmtree(klasor_yolu)

# 2. EMBEDDİNG MODELİ (Türkçe ve Çok Dilli Destekli)
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

# 3. VERİ OKUMA VE AKILLI BÖLME
print("Standartlaştırılmış metin dosyası okunuyor...")
try:
    with open("hastaliklar.txt", "r", encoding="utf-8") as f:
        icerik = f.read()

    # Markdown başlıklarına göre (# ve ##) bölüyoruz
    headers_to_split_on = [
        ("#", "Kategori"), 
        ("##", "Hastalik_Adi")
    ]
    
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    dokumanlar = markdown_splitter.split_text(icerik)
    
    # İçeriği zenginleştirme: Arama motorunun daha iyi bulması için metadata'yı içeriğe ekliyoruz
    for doc in dokumanlar:
        h_adi = doc.metadata.get("Hastalik_Adi", "Bilinmeyen")
        kat = doc.metadata.get("Kategori", "Bilinmeyen")
        
        # İçeriği: Kategori + Hastalık Adı + (Belirtiler + Anahtar Kelimeler + Protokol) şeklinde birleştiriyoruz
        doc.page_content = f"Kategori: {kat}\nHastalık: {h_adi}\n{doc.page_content}"

except FileNotFoundError:
    print("HATA: 'hastaliklar.txt' dosyası bulunamadı! Lütfen dosya ismini kontrol edin.")
    exit()

# 4. CHROMA VERİTABANINA KAYIT (Cosine Similarity ile)
print(f"{len(dokumanlar)} adet hastalık vektörlere dönüştürülüyor...")
vectorstore = Chroma.from_documents(
    documents=dokumanlar,
    embedding=embeddings,
    persist_directory=klasor_yolu,
    collection_metadata={"hnsw:space": "cosine"} # Benzerlik ölçümü için en hassas yöntem
)

print(f"BAŞARILI: Veritabanı '{klasor_yolu}' klasörüne kaydedildi.")