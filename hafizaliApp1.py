import os
import langchain # Sistem-Gözlem-Ayarları için eklendi
from langchain_ollama import ChatOllama, OllamaEmbeddings
# DÜZELTME: Chroma artık eski topluluk (community) paketinden değil, kendi güncel paketinden çağrılıyor.
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
print("başlıyor...")
# Sistem-Gözlem-Ayarları: Arka planda dönen tüm işlemleri terminale yazdırır
langchain.debug = True

# 1. VERİ YÜKLEME VE HAFIZA BAĞLANTISI
embeddings = OllamaEmbeddings(
    model="nomic-embed-text",
    base_url="http://host.docker.internal:11434"
)
persist_directory = "./chroma_db"

vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

#kontrol
print("\n--- Sistem-Gözlem-Ayarları: LangChain Debug Aktif ---\n")
# ... (Mevcut kodunuzun üst kısmı)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# --- VERİTABANI KONTROL TESTİ ---
print("\n[TEST] Veritabanı kontrol ediliyor...")

# 1. Kayıt Sayısını Kontrol Et
# Veritabanında kaç adet döküman (parça) olduğunu yazdırır.
kayit_sayisi = vectorstore._collection.count()
print(f"Veritabanındaki toplam parça sayısı: {kayit_sayisi}")

if kayit_sayisi == 0:
    print("UYARI: Veritabanı boş görünüyor! Lütfen veri yükleme işlemini kontrol edin.")
else:
    # 2. Örnek Bir Arama Yap
    # Veritabanının gerçekten ilgili içeriği getirip getirmediğini test eder.
    test_sorusu = "baş ağrısı" # Verilerinizde olan bir kelimeyi buraya yazın
    test_sonuclari = vectorstore.similarity_search(test_sorusu, k=1)
    
    if test_sonuclari:
        print(f"Test Araması Başarılı! İlk eşleşen içerik: {test_sonuclari[0].page_content[:100]}...")
    else:
        print("UYARI: Arama yapıldı ama sonuç dönmedi. Embedding modelinde sorun olabilir.")
print("--- TEST BİTTİ ---\n")
print()

# ... (Mevcut kodunuzun devamı: 2. GELİŞMİŞ SİSTEM TALİMATI...)

# 2. GELİŞMİŞ SİSTEM TALİMATI (Hafıza Destekli)
sablon = """
Sen uzman, empatik ve profesyonel bir doktorsun. Bir asistan veya yapay zeka olduğunu ASLA belli etme. İçsesini veya ne yapacağını (Örn: "mesajını okudum", "soruma cevap ver") ASLA hastaya söyleme. Doğrudan insan insana iletişim kur.

BİLGİ KAYNAĞI:
{context}

GEÇMİŞ KONUŞMA:
{history}

KESİN KURALLAR:
1. Bilgi kaynağındaki hastalıklardan hastanın şikayetine en uygun olanı seç.
2. DİKKAT - SORU SORMA ŞARTI: Eğer hastanın verdiği bilgiler protokoldeki şartları henüz SAĞLAMIYORSA, eksik bilgiyi öğrenmek için sadece o hastalığın 'Protokol' kısmında yazan soruyu doğal bir dille sor.
3. DİKKAT - TANI KOYMA ŞARTI: Eğer hasta 'Protokol' kısmındaki soruya zaten cevap verdiyse ve şartlar SAĞLANDIYSA, asla başka soru sorma! Doğrudan tanıyı koy (Örn: "Bu belirtiler ışığında ... tanısı koyuyorum") ve nedenini açıkla.

Hasta: {question}
Doktor:"""

PROMPT = ChatPromptTemplate.from_template(sablon)

# 3. SİSTEMİ BİRLEŞTİRME (LCEL)
llm = ChatOllama(
    model="llama3", 
    temperature=0.1,
    base_url="http://host.docker.internal:11434"
) 

demo_ephemeral_chat_history = ChatMessageHistory()

# SİSTEM DÜZELTMESİ: Python objelerini düz metne (Senaryoya) çeviren fonksiyon
def gecmisi_metne_cevir(mesajlar):
    okunabilir_metin = ""
    for mesaj in mesajlar:
        if mesaj.type == "human":
            okunabilir_metin += f"Hasta: {mesaj.content}\n"
        else:
            okunabilir_metin += f"Doktor: {mesaj.content}\n"
    return okunabilir_metin

# Güncellenmiş Zincir (Chain)
qa_chain = (
    {
        "context": retriever, 
        "question": RunnablePassthrough(),
        # BURASI DEĞİŞTİ: Artık ham obje değil, temiz metin gidiyor
        "history": lambda x: gecmisi_metne_cevir(demo_ephemeral_chat_history.messages)
    }
    | PROMPT
    | llm
    | StrOutputParser()
)

# 4. ÇALIŞTIRMA DÖNGÜSÜ
print("\n--- Klinik Asistan Sistem-Başlatıldı (Hafızalı ve Gözlem Modu) ---")
while True:
    query = input("Siz: ")
    if query.lower() in ["çıkış", "exit"]: break
    
    result = qa_chain.invoke(query)
    
    demo_ephemeral_chat_history.add_user_message(query)
    demo_ephemeral_chat_history.add_ai_message(result)
    
    print(f"\nAsistan: {result}\n")