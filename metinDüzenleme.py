from langchain_text_splitters import MarkdownHeaderTextSplitter

# 1. BAŞLIK AYARLARI: Sisteme hangi işaretin ne anlama geldiğini öğretiyoruz.
baslik_ayarlari = [
    ("#", "Bolum"),      # Tek kare işareti 'Bolum' (Örn: ÜROLOJİ) bilgisini tutar
    ("##", "Hastalik")   # Çift kare işareti 'Hastalik' (Örn: Böbrek Taşı) bilgisini tutar
]

# 2. METİN BÖLÜCÜ KURULUMU
metin_bolucu = MarkdownHeaderTextSplitter(headers_to_split_on=baslik_ayarlari)

# 3. DOSYAYI OKUMA (Yukarıda oluşturduğun hastaliklar_md.txt dosyasını okuyoruz)
with open("hastaliklar.txt", "r", encoding="utf-8") as dosya:
    okunan_metin = dosya.read()

# 4. BÖLME İŞLEMİ
ayrilmis_metinler = metin_bolucu.split_text(okunan_metin)

# ---------------------------------------------------------
# 5. TEST VE DOĞRULAMA KODU (DEBUG)
# Sistemin metni doğru bölüp bölmediğini kontrol ediyoruz.
# ---------------------------------------------------------

print("\n=== SİSTEM-BAŞLAT-AYARLAR: VERİ BÖLME TESTİ ===\n")
print(f"Toplam tespit edilen hastalık (parça) sayısı: {len(ayrilmis_metinler)}\n")

# Hepsini yazdırırsak ekran çok dolar, o yüzden sadece ilk 2 ve son 1 parçayı test edelim.
test_edilecek_parcalar = [ayrilmis_metinler[0], ayrilmis_metinler[1], ayrilmis_metinler[-1]]

for i, parca in enumerate(test_edilecek_parcalar):
    print(f"--- TEST PARÇASI {i+1} ---")
    print(f"Ana Bölüm (Kategori): {parca.metadata.get('Bolum')}")
    print(f"İlgili Hastalık     : {parca.metadata.get('Hastalik')}")
    print(f"\nYapay Zekanın Göreceği İçerik:\n{parca.page_content}")
    print("-" * 40 + "\n")

