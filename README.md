# Yapay Zeka Destekli Klinik Asistanı

Bu proje, Büyük Dil Modelleri (LLM) kullanarak hastane triyaj ve tanı süreçlerine destek olmak amacıyla geliştirilmiştir. Docker desteği sayesinde kurulum karmaşası olmadan her ortamda hızlıca çalıştırılabilir.

## 🛠️ Kurulum ve Çalıştırma

Projeyi kendi bilgisayarınızda çalıştırmak için aşağıdaki adımları izleyin.

### 1. Ön Gereksinimler
Sisteminizde **Docker** kurulu olmalıdır.

### 2. Projeyi Klonlayın
```bash
git clone [https://github.com/TunahanYlcnn/Yapay-Zeka-Destekli-Klinik-Asistan-.git](https://github.com/TunahanYlcnn/Yapay-Zeka-Destekli-Klinik-Asistan-.git)
cd Yapay-Zeka-Destekli-Klinik-Asistan-
```

### 3. Docker İmajını Oluşturun
Proje klasöründeyken terminale şu komutu yazarak Docker imajını hazırlayın:
```bash
docker-compose up --build -d
```

## 📦 Proje İçeriği ve Yapılandırma
Docker Kullanımı: Proje tamamen Dockerize edilmiştir. Dockerfile üzerinden tüm sistem bağımlılıkları otomatik yüklenir.
Python Bağımlılıkları: Gerekli tüm paketler requirements.txt dosyasında listelenmiştir.

