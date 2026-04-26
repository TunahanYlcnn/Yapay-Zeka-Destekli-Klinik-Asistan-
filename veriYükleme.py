# Dosya İsmi: veriYükleme.py
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings 
from langchain_core.documents import Document
import os
import shutil

klasor_yolu = "./chroma_db"
if os.path.exists(klasor_yolu):
    shutil.rmtree(klasor_yolu)

# TÜRKÇE İÇİN EN İYİ MODEL: 
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

dokumanlar = []
try:
    with open("hastaliklar.txt", "r", encoding="utf-8") as dosya:
        icerik = dosya.read()
        # Her hastalığı '## Hastalık:' ifadesinden bölüyoruz
        parcalar = icerik.split("## Hastalık:")
        
        for parca in parcalar:
            temiz_parca = parca.strip()
            if temiz_parca and "Belirtiler:" in temiz_parca:
                # Başına ekstra 'Hastalık:' eklemiyoruz, metnin orijinalini koruyoruz
                dokumanlar.append(Document(page_content=f"## Hastalık: {temiz_parca}"))

    vectorstore = Chroma.from_documents(
        documents=dokumanlar,
        embedding=embeddings,
        persist_directory=klasor_yolu,
        collection_metadata={"hnsw:space": "cosine"} # Mesafeyi normalize eder
    )
    print(f"Başarıyla {len(dokumanlar)} hastalık kaydedildi.")
except Exception as e:
    print(f"Hata: {e}")
