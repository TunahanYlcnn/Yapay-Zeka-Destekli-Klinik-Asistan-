# Dosya İsmi: test.py

from langchain_chroma import Chroma
# Sistem-Güncelleme: Kütüphane uyarısını kaldırmak için en güncel paket eklendi
from langchain_huggingface import HuggingFaceEmbeddings 

print("Sistem-Gözlem-Ayarları: Test başlatılıyor...")

# 1. DOĞRU MODELİ AYARLAMA (Kayıt yaparken kullandığımız modelin aynısı olmak ZORUNDA!)
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

# 2. VERİTABANINA BAĞLANMA
klasor_yolu = "./chroma_db"
vectorstore = Chroma(persist_directory=klasor_yolu, embedding_function=embeddings)

# 3. KAYIT SAYISINI KONTROL ETME
kayit_sayisi = vectorstore._collection.count()
print(f"Veritabanındaki toplam hastalık (döküman) sayısı: {kayit_sayisi}")

# Eğer veritabanı boş değilse arama testine geçiyoruz
if kayit_sayisi > 0:
    
    while True:
        test_sorusu = input("\nTest Sorusu Giriniz (Çıkmak için 'q' yazın): ")
        if test_sorusu.lower() == 'q':
            break
            
        # ANLAMSAL ARAMA (SEMANTIC SEARCH) TESTİ
        sonuclar = vectorstore.similarity_search(test_sorusu, k=1)
        
        print("\n--- Sistem Tarafından Bulunan En Yakın Hastalık ---")
        print(sonuclar[0].page_content)
        print("---------------------------------------------------")

else:
    print("Sistem-Hatası: Veritabanı boş görünüyor.")