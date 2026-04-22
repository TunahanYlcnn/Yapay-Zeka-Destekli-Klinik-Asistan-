import os
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
# 2026 v1.x standartları: Artık chains modülüne ihtiyacımız yok!
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# 1. VERİ YÜKLEME VE İŞLEME
loader = TextLoader("hastaliklar.txt", encoding="utf-8")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=600, chunk_overlap=100)
texts = text_splitter.split_documents(documents)

# 2. YEREL VE ÜCRETSİZ BEYİN (Ollama)
embeddings = OllamaEmbeddings(
    model="llama3",
    base_url="http://host.docker.internal:11434"
)
persist_directory = "./chroma_db" # Veritabanının kaydedileceği klasör

if os.path.exists(persist_directory):
    # Eğer klasör varsa baştan oluşturma, kayıtlı olanı yükle
    vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
else:
    # İlk çalıştırmada oluştur ve kaydet
    vectorstore = Chroma.from_documents(texts, embeddings, persist_directory=persist_directory)
# Veri getirme mekanizması (retriever)
retriever = vectorstore.as_retriever()

# 3. ÖZEL SİSTEM TALİMATI (Prompt Engineering)
sablon = """
Sen profesyonel bir Klinik Asistanısın. Aşağıdaki bilgilere göre hareket et:
{context}

KURAL: 
1. Eğer hastanın verdiği bilgiler metindeki 'Protokol' kısmındaki şartları tam karşılamıyorsa, kesinlikle tanı koyma ve eksik bilgiyi öğrenmek için 'Protokol'deki soruyu sor.
2. Eğer hasta tüm şartları sağlıyorsa, tanıyı koy ve nedenini açıkla.
3. Ne olursa olsun HER ZAMAN Türkçe dilinde konuş ve yanıt ver.


Hasta Mesajı: {question}
Asistan Yanıtı:"""

PROMPT = ChatPromptTemplate.from_template(sablon)

# 4. SİSTEMİ BİRLEŞTİRME (LCEL Yapısı)
# Bu yapı doğrudan kütüphane bağımlılığı hatalarını baypas eder.
llm = ChatOllama(
    model="llama3", 
    temperature=0.1,
    base_url="http://host.docker.internal:11434"
) 

# Modern Zincirleme: Girdiyi al -> Dokümanı bul -> Prompta yerleştir -> LLM'e gönder -> Metne çevir
# Değişken ismini 'qa_chain' olarak koruyoruz.
qa_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | PROMPT
    | llm
    | StrOutputParser()
)

# 5. ÇALIŞTIRMA DÖNGÜSÜ
print("\n--- Klinik Asistan Sistem-Başlatıldı (Modern LCEL Modu) ---")
while True:
    query = input("Şikayetini yaz: ")
    if query.lower() in ["çıkış", "exit"]: break
    
    # LCEL doğrudan string döndürdüğü için 'answer' anahtarına gerek kalmaz
    result = qa_chain.invoke(query)
    print(f"\nAsistan: {result}\n")