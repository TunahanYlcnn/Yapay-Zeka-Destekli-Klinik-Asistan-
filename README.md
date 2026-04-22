\# LLM Giriş Projesi



Bu proje, Büyük Dil Modelleri (LLM) üzerine çalışmalar yapmak için oluşturulmuş temel bir geliştirme ortamıdır. Proje, Docker kullanılarak konteynerize edilmiştir; bu sayede bağımlılıklarla uğraşmadan her ortamda hızlıca çalıştırılabilir.



\## 🛠️ Kurulum ve Çalıştırma



Projeyi kendi bilgisayarınızda çalıştırmak için aşağıdaki adımları izleyin.



\### 1. Ön Gereksinimler

Sisteminizde \*\*Docker\*\* kurulu olmalıdır.



\### 2. Projeyi Kopyalayın

```bash

git clone \[https://github.com/tunahan/llmGiris.git](https://github.com/tunahan/llmGiris.git)

cd llmGiris



\### 3. Docker İmajını Oluşturun

Proje klasöründeyken terminale şu komutu yazarak Docker imajını hazırlayın:



docker build -t llm-giris-projesi .



\###4. Konteyneri Çalıştırın

İmaj oluştuktan sonra projeyi şu komutla ayağa kaldırın:



docker run -it --name llm-calisma-alani llm-giris-projesi



📦 Proje İçeriği ve Yapılandırma

Docker Kullanımı: Proje tamamen Dockerize edilmiştir. Dockerfile üzerinden tüm sistem bağımlılıkları otomatik yüklenir.



Python Bağımlılıkları: Gerekli tüm paketler requirements.txt dosyasında listelenmiştir.



Geliştirme Ortamı: Proje Windows 11 Pro üzerinde, Lenovo Gaming sisteminde geliştirilmiştir.





