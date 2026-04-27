import os
from langchain_ollama import ChatOllama
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_message_histories import ChatMessageHistory
from sentence_transformers import CrossEncoder

#device = "cpu" # eğer gpu patlarsa değiştir

# 1. SETUP & MODELS
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    model_kwargs={'device': "cuda"}
)
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
reranker_model = CrossEncoder('BAAI/bge-reranker-v2-m3', device="cpu", trust_remote_code=True)

#modeller
#llm = ChatOllama(model="llama3", temperature=0, base_url="http://host.docker.internal:11434")
llm = ChatOllama(model="gemma2:2b", temperature=0, base_url="http://host.docker.internal:11434")

# 2. SEARCH QUERY GENERATOR (English Instructions for better reasoning)
search_prompt = ChatPromptTemplate.from_template("""
Carefully analyze the symptoms and location details (heel, ankle, front, back, etc.) in the patient’s latest message.
If the patient rejects previous suggestions and provides a new symptom (“morning pain,” “stiffness,” etc.), 
formulate the search query entirely based on these new symptoms. 

History: {history}
Patient's Last Message: {question}

TASK: Generate 3-4 essential Turkish medical keywords based on actual symptoms.
Search Query (Turkish Only):""")

search_query_chain = search_prompt | llm | StrOutputParser()

# 3. DOCTOR PERSONA (English System Prompt - Turkish Response)
# DÜZELTME: Modelin dökümanı kopyalaması engellendi ve kıyaslama yeteneği eklendi.
doctor_prompt = ChatPromptTemplate.from_template("""
SYSTEM INSTRUCTION:
You are an expert, direct, and professional medical doctor. 
CRITICAL RULE: Always respond in TURKISH. Never use English in your final output.

CONTEXT INFORMATION (Multiple Potential Protocols):
{context}

CONVERSATION HISTORY:
{history}

STRICT RULES:
1. ACT, DON'T READ: Do NOT quote the protocol text literally. Never say "Protokol şunu diyor" or "Soru sor: X". Instead, ask the question directly as a doctor (e.g., "Ateşiniz var mı?").
2. COMPARE CONTEXTS: You are provided with multiple disease protocols. Compare them. If the patient mentions 'foamy urine', look for the disease that specifically includes 'foamy urine' (Nefrotik Sendrom), even if other diseases share different keywords.
3. EXTREME BREVITY: Do not repeat symptoms. Ask ONLY the next necessary question from the protocol.
4. DIAGNOSIS: If all conditions for a specific disease are met, say "[Hastalık Adı] tanısını koyuyorum." and give a 1-sentence explanation.
5. NO HALLUCINATIONS: If the patient says they DON'T have a symptom (like infection), move to the next logical protocol in the provided context.

Patient: {question}
Doctor (Response MUST be in Turkish):""")

# 4. HELPER FUNCTIONS
demo_history = ChatMessageHistory()

def get_history(_):
    messages = demo_history.messages
    readable = ""
    for m in messages:
        prefix = "Hasta: " if m.type == "human" else "Doktor: "
        readable += f"{prefix}{m.content}\n"
    return readable

def rerank_logic(inputs):
    # Smart query generation
    smart_query = search_query_chain.invoke({"history": get_history(None), "question": inputs["question"]})
    
    docs = retriever.invoke(smart_query)
    if not docs: return "Bilgi bulunamadı."
    
    # Cross-encoder re-ranking
    pairs = [[smart_query, doc.page_content] for doc in docs]
    scores = reranker_model.predict(pairs)
    
    # DÜZELTME: Modelin kıyas yapabilmesi için en iyi 3 dökümanı gönderiyoruz.
    # Bu, 'balon' kelimesiyle yanlış dökümana (Üreterosel) saplanmasını engeller.
    scored_docs = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
    
    # Top 3 dökümanı aralarına belirgin ayraçlar koyarak birleştiriyoruz
    context_text = ""
    for i, (score, doc) in enumerate(scored_docs[:3]):
        context_text += f"--- POTENTIAL DISEASE {i+1} ---\n{doc.page_content}\n\n"
        
    return context_text

# 5. CHAIN ASSEMBLY
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

# 6. DIALOG LOOP
print("\n--- Klinik Asistan Başlatıldı (Multi-Context Mode) ---")
while True:
    user_in = input("Siz: ").encode('utf-8', 'ignore').decode('utf-8')
    if user_in.lower() in ["q", "çıkış"]: break
    
    # Önce kullanıcı mesajını ekle (Model bunu görsün)
    demo_history.add_user_message(user_in)

    # Sonra modeli çağır
    response = full_chain.invoke({"question": user_in})

    # En son asistanın cevabını ekle
    demo_history.add_ai_message(response)
    
    print(f"\nAsistan: {response}\n")