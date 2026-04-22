# 1. Temel imaj olarak resmi Python imajını kullanıyoruz
FROM python:3.11-slim

# 2. Konteyner içindeki çalışma dizinini ayarlıyoruz
# Bu klasör konteynerin içinde oluşturulacak 'ana klasör'dür
WORKDIR /llmGiris

# 3. Gereksinim dosyasını çalışma dizinine kopyalıyoruz
COPY requirements.txt .

# 4. Bağımlılıkları yüklüyoruz
# --no-cache-dir seçeneği gereksiz dosyaları temiz tutarak imajı hafifletir
RUN pip install --no-cache-dir -r requirements.txt

# 5. Projenin geri kalan dosyalarını konteyner içine kopyalıyoruz
# (hastaliklar.txt ve klinik_rag.py dosyaların burada kopyalanır)
COPY . .

# 7. Konteyner başlatıldığında çalıştırılacak komut
# Buradaki 'klinik_rag.py' isminin senin Python dosyanla aynı olduğundan emin ol
CMD ["python", "app.py"]