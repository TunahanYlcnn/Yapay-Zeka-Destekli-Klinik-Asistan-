from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

print("Sistem-Gözlem-Ayarları: Test başlatılıyor...")

# 1. DOĞRU MODELİ AYARLAMA (Kayıt yaparken kullandığımız modelin aynısı olmak ZORUNDA!)
embeddings = OllamaEmbeddings(
    model="nomic-embed-text",
    base_url="http://host.docker.internal:11434"
)

# 2. VERİTABANINA BAĞLANMA
klasor_yolu = "./chroma_db"
vectorstore = Chroma(persist_directory=klasor_yolu, embedding_function=embeddings)

# 3. KAYIT SAYISINI KONTROL ETME
kayit_sayisi = vectorstore._collection.count()
print(f"Veritabanındaki toplam hastalık (döküman) sayısı: {kayit_sayisi}")

# Eğer veritabanı boş değilse arama testine geçiyoruz
if kayit_sayisi > 0:
    # 4. ANLAMSAL ARAMA (SEMANTIC SEARCH) TESTİ
    # Burada txt dosyasındaki birebir aynı kelimeleri DEĞİL, benzer anlamda bir şikayet yazıyoruz.
    # Amacımız modelin kelimeleri değil, "anlamı" kavrayıp kavramadığını görmek.
    
    while True:
        test_sorusu = input("\nTest Sorusu Giriniz (Çıkmak için 'q' yazın): ")
        if test_sorusu.lower() == 'q':
            break
        
        # k=1 diyerek veritabanından en yüksek benzerliğe sahip 1 sonucu getirmesini istiyoruz.
        sonuclar = vectorstore.similarity_search(test_sorusu, k=3)
        
        print("\n--- Sistem Tarafından Bulunan En Yakın Hastalık ---")
        print(sonuclar[0].page_content)
        print(sonuclar[1].page_content)
        print(sonuclar[2].page_content)
        print("---------------------------------------------------")
    
else:
    print("Sistem-Hatası: Veritabanı boş görünüyor.")