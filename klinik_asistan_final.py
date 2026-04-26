import os
from langchain_ollama import ChatOllama
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_message_histories import ChatMessageHistory
from sentence_transformers import CrossEncoder

# 1. KURULUM VE MODELLER
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
reranker_model = CrossEncoder('BAAI/bge-reranker-v2-m3', trust_remote_code=True)

llm = ChatOllama(model="llama3", temperature=0, base_url="http://host.docker.internal:11434")

# 2. SORGUNU YENİDEN YAZICI (Search Query Generator)
# Bu kısım hastanın "Götüremiyorum" gibi kısa cevaplarını anlamlı arama terimlerine dönüştürür.
search_prompt = ChatPromptTemplate.from_template("""
Hastanın son mesajındaki belirtileri ve konum bilgilerini (topuk, bilek, ön, arka vb.) dikkatlice analiz et.
Eğer hasta önceki tahminleri reddedip yeni bir belirti ("sabah ağrısı", "kilitlenme" vb.) veriyorsa, 
arama sorgusunu tamamen bu yeni belirtilere göre oluştur. 

Geçmiş: {history}
Hasta Son Cevap: {question}
Arama Sorgusu (Sadece en önemli 2-3 tıbbi anahtar kelimeyi Türkçe yaz):""")

search_query_chain = search_prompt | llm | StrOutputParser()

# 3. ANA DOKTOR PROMPT (Türkçe Kilidi ve Persona)
doctor_prompt = ChatPromptTemplate.from_template("""
KESİN KURAL: SADECE TÜRKÇE KONUŞACAKSIN. ASLA İNGİLİZCE KELİME KULLANMA.
Sen uzman bir doktorsun. Bir yapay zeka olduğunu belli etme.

PROTOKOL BİLGİSİ:
{context}

GEÇMİŞ:
{history}

TALİMAT:
1. Protokoldeki şartlar (sorular) tamamlanmadan tanı koyma.
2. Eğer şartlar sağlandıysa doğrudan tanıyı söyle ve nedenini açıkla.
3. Hastaya karşı nazik ve profesyonel ol.

Hasta: {question}
Doktor (Sadece Türkçe):""")

# 4. YARDIMCI FONKSİYONLAR
demo_history = ChatMessageHistory()

def get_history(_):
    messages = demo_history.messages
    okunabilir = ""
    for m in messages:
        okunabilir += f"{'Hasta' if m.type == 'human' else 'Doktor'}: {m.content}\n"
    return okunabilir

def rerank_logic(inputs):
    smart_query = search_query_chain.invoke({"history": get_history(None), "question": inputs["question"]})
    print(f"\n[SİSTEM] Akıllı Arama Terimi: {smart_query}")
    
    docs = retriever.invoke(smart_query)
    if not docs: return "Bilgi bulunamadı."
    
    pairs = [[smart_query, doc.page_content] for doc in docs]
    scores = reranker_model.predict(pairs)
    
    # NOKTA ATIŞI DÜZELTMESİ: Sadece en iyiyi değil, 
    # en yüksek puanlı ilk 2 veya 3 dökümanı gönderelim ki model kıyas yapabilsin.
    scored_docs = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
    best_docs_content = "\n\n".join([d.page_content for s, d in scored_docs[:2]])
    
    return best_docs_content

# 5. ZİNCİRİN BİRLEŞTİRİLMESİ
full_chain = (
    {
        "context": RunnableLambda(rerank_logic),
        "question": RunnablePassthrough(),
        "history": RunnableLambda(get_history)
    }
    | doctor_prompt
    | llm
    | StrOutputParser()
)

# 6. DÖNGÜ
print("\n--- Gelişmiş Klinik Asistan Başlatıldı ---")
while True:
    user_in = input("Siz: ")
    if user_in.lower() in ["q", "çıkış"]: break
    
    # Yanıt al
    response = full_chain.invoke({"question": user_in})
    
    # Hafızayı güncelle
    demo_history.add_user_message(user_in)
    demo_history.add_ai_message(response)
    
    print(f"\nAsistan: {response}\n")