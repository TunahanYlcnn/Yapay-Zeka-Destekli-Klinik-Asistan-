import os
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
print("başlıyor...")
# 1. HAFIZAYI YÜKLE (Mevcut chroma_db klasörünü okuyoruz)
embeddings = OllamaEmbeddings(
    model="nomic-embed-text",
    base_url="http://host.docker.internal:11434"
)
persist_directory = "./chroma_db"

# Veritabanını baştan oluşturmuyoruz, sadece bağlanıyoruz
vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3}) # En alakalı 3 hastalığı getir

# 2. SİSTEM TALİMATI (Personas)
sablon = """
Sen profesyonel bir Klinik Asistanısın. Aşağıdaki tıbbi protokol metnine bağlı kalarak yanıt ver:
{context}

Kritik Kurallar:
1. Eğer hastanın şikayeti protokoldeki 'Belirtiler' ile uyuşuyorsa, hemen tanı koyma! Önce 'Protokol' kısmındaki soruyu sor.
2. Eğer hasta protokoldeki tüm şartları sağlıyorsa (sorularına 'evet' cevabı verdiyse), o zaman tanıyı koy.
3. Eğer bilgi metinde yoksa, 'Bu konuda bilgim yok, lütfen bir uzmana danışın' de.
4. HER ZAMAN Türkçe yanıt ver.

Hasta Mesajı: {question}
Asistan Yanıtı:"""

PROMPT = ChatPromptTemplate.from_template(sablon)

# 3. ZİNCİRİ KUR (LCEL Yapısı)
llm = ChatOllama(
    model="llama3", 
    temperature=0.1, # Daha tutarlı ve ciddi cevaplar için düşük sıcaklık
    base_url="http://host.docker.internal:11434"
) 

qa_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | PROMPT
    | llm
    | StrOutputParser()
)

# 4. ÇALIŞTIRMA
print("\n--- Klinik Asistan Sistem-Başlatıldı ---")
while True:
    query = input("Şikayetiniz nedir? (Çıkmak için 'exit'): ")
    if query.lower() in ["çıkış", "exit"]: break
    
    result = qa_chain.invoke(query)
    print(f"\nAsistan: {result}\n")