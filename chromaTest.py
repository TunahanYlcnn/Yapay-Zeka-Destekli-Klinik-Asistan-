# Dosya İsmi: chromeTest.py
import os
import time
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder

print("--- [ADIM 2] Re-ranker Destekli Nokta Atışı Testi ---")

# 1. VERİTABANI BAĞLANTISI
if not os.path.exists("./chroma_db"):
    print("HATA: Veritabanı bulunamadı. Lütfen önce veriYükleme.py dosyasını çalıştırın.")
    exit()

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

# 2. RE-RANKER MODELİ YÜKLEME (GÜNCELLENDİ)
print("Zeki sıralayıcı (Re-ranker) yükleniyor, lütfen bekleyin...")
# NOT: 'BAAI/bge-reranker-v2-m3' modeli Türkçe performansında mükemmeldir.
try:
    # Alternatif olarak 'cornetto/mmarco-mMiniLM-L12-v2' de kullanılabilir
    reranker = CrossEncoder('BAAI/bge-reranker-v2-m3', trust_remote_code=True)
except Exception as e:
    print(f"Model yükleme hatası: {e}")
    print("İnternet bağlantınızı veya model ismini kontrol edin.")
    exit()

def nokta_atisi_arama(soru, k_aday=10):
    # A. İlk aşama: Vektör araması ile adayları getir
    adaylar = vectorstore.similarity_search(soru, k=k_aday)
    
    if not adaylar:
        return []

    # B. İkinci aşama: Re-ranker ile adayları puanla
    pairs = [[soru, doc.page_content] for doc in adaylar]
    puanlar = reranker.predict(pairs)
    
    # C. Puanları eşleştir ve sırala
    sirali_sonuclar = sorted(zip(puanlar, adaylar), key=lambda x: x[0], reverse=True)
    return sirali_sonuclar

# 3. TEST DÖNGÜSÜ
print("\nSistem hazır. Sorgu yapabilirsiniz.")

while True:
    test_sorusu = input("\nHasta Şikayeti (Çıkış için 'q'): ")
    if test_sorusu.lower() == 'q':
        break
    
    start_t = time.time()
    analiz_sonuclari = nokta_atisi_arama(test_sorusu)
    sure = time.time() - start_t

    print(f"\n--- Analiz Tamamlandı ({sure:.2f} saniye) ---")
    
    for i, (puan, doc) in enumerate(analiz_sonuclari[:3]):
        # BGE Reranker skorları genelde farklı skalalarda olabilir, sıralama esastır.
        print(f"{i+1}. SIRADAKİ EŞLEŞME (Re-rank Puanı: {puan:.4f})")
        print(f"{doc.page_content}")
        print("-" * 60)