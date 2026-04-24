import os
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
# Hafıza için gerekli yeni kütüphaneler
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# 1. VERİ YÜKLEME VE HAFIZA BAĞLANTISI
embeddings = OllamaEmbeddings(
    model="nomic-embed-text",
    base_url="http://host.docker.internal:11434"
)
persist_directory = "./chroma_db"

vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# 2. GELİŞMİŞ SİSTEM TALİMATI (Hafıza Destekli)
# ChatPromptTemplate içinde 'history' için bir yer ayırıyoruz
sablon = """
Sen profesyonel bir Klinik Asistanısın. Aşağıdaki bilgilere ve geçmiş konuşmalara göre hareket et:

BİLGİ KAYNAĞI (DOKÜMANLAR):
{context}

GEÇMİŞ KONUŞMA ÖZETİ:
{history}

KURALLAR:
1. Eğer hastanın verdiği bilgiler protokoldeki şartları tam karşılamıyorsa, kesinlikle tanı koyma. Eksik bilgiyi öğrenmek için protokoldeki soruyu sor.
2. Eğer hasta geçmiş mesajlarında bu soruyu zaten cevapladıysa, bunu hatırla ve bir sonraki adıma geç veya tanıyı koy.
3. Her zaman profesyonel ve yardımcı bir ton kullan.
4. HER ZAMAN Türkçe konuş.

Hasta Mesajı: {question}
Asistan Yanıtı:"""

PROMPT = ChatPromptTemplate.from_template(sablon)

# 3. SİSTEMİ BİRLEŞTİRME (LCEL)
llm = ChatOllama(
    model="llama3", 
    temperature=0.1,
    base_url="http://host.docker.internal:11434"
) 

# Hafızayı saklayacağımız nesne
demo_ephemeral_chat_history = ChatMessageHistory()

# Ana zincir (Chain)
qa_chain = (
    {
        "context": retriever, 
        "question": RunnablePassthrough(),
        "history": lambda x: demo_ephemeral_chat_history.messages # Geçmiş mesajları buraya enjekte eder
    }
    | PROMPT
    | llm
    | StrOutputParser()
)

# 4. ÇALIŞTIRMA DÖNGÜSÜ
print("\n--- Klinik Asistan Sistem-Başlatıldı (Hafızalı Mod) ---")
while True:
    query = input("Siz: ")
    if query.lower() in ["çıkış", "exit"]: break
    
    # Yanıtı üret
    result = qa_chain.invoke(query)
    
    # ÖNEMLİ: Konuşmayı hafızaya kaydet
    demo_ephemeral_chat_history.add_user_message(query)
    demo_ephemeral_chat_history.add_ai_message(result)
    
    print(f"\nAsistan: {result}\n")