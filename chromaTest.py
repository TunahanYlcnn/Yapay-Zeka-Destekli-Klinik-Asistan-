from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings 

# Model aynı olmak ZORUNDA
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
    # 4. ANLAMSAL ARAMA (SEMANTIC SEARCH) TESTİ
    # Burada txt dosyasındaki birebir aynı kelimeleri DEĞİL, benzer anlamda bir şikayet yazıyoruz.
    # Amacımız modelin kelimeleri değil, "anlamı" kavrayıp kavramadığını görmek.
    
    while True:
        test_sorusu = input("\nTest Sorusu Giriniz (Çıkmak için 'q' yazın): ")
        if test_sorusu.lower() == 'q':
            break
        
        # k=3 diyerek veritabanından en yüksek benzerliğe sahip 3 sonucu getirmesini istiyoruz.
        sonuclar = vectorstore.similarity_search_with_score(test_sorusu, k=3)
        
        print("\n--- Sistem Tarafından Bulunan En Yakın 3 Hastalık ---")
        for i, (doc, score) in enumerate(sonuclar, 1):
            print(f"{i}. Bulunan: {doc.page_content}\n   Güven Skoru (Düşük olması daha iyidir): {score}\n")
        print("---------------------------------------------------")
    
else:
    print("Sistem-Hatası: Veritabanı boş görünüyor.")