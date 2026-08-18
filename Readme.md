# Qwen IoT Lab Assistant

Local LLM + RAG tabanlı yapay zekâ asistanı.

Bu proje **Qwen** modeli, **FastAPI**, **Qdrant** ve multilingual embedding modeli kullanarak yerel olarak çalışan bir yapay zekâ servisi sağlar.

Model ağırlıkları büyük olduğu için GitHub repository içerisinde bulunmaz.  
Projeyi indirdikten sonra model dosyalarının ayrıca indirilmesi gerekir.

---
## Proje Yapısı

Projenin ana dizin yapısı aşağıdaki gibidir:

```text
Files/
│
├── rag/
│   │
│   ├── backups/
│   ├── evaluation/
│   ├── incident_detection/
│   ├── log_agent/
│   ├── qdrant/
│   ├── scripts/
│   ├── training/
│   ├── training_management/
│   ├── training_pipeline/
│   ├── worker/
│   ├── export_training_dataset.py
│   └── migrate_telemetry_v2.py
│
├── tools/
│   └── whichllm
│
├── .gitignore
└── Readme.md

# Gereksinimler

Projeyi çalıştırmak için aşağıdaki araçların sisteminizde kurulu olması gerekir:

- Python 3.10+
- Git
- Docker
- Docker Compose
- En az 8 GB RAM
- Önerilen: 16 GB+ RAM

GPU zorunlu değildir ancak model performansı için önerilir.

Linux için:

```bash
python3 --version
git --version
docker --version
```

Windows için:

```powershell
python --version
git --version
docker --version
```

---

# 1. Repository'yi İndirme

Terminal açın.

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

Ardından proje klasörüne girin:

```bash
cd huggingface-model-server
```

Örnek:

```bash
git clone https://github.com/USERNAME/huggingface-model-server.git
cd huggingface-model-server
```

---

# 2. Python Virtual Environment Oluşturma

## Linux / macOS

```bash
python3 -m venv .venv
```

Aktifleştirin:

```bash
source .venv/bin/activate
```

---

## Windows PowerShell

```powershell
python -m venv .venv
```

Aktifleştirin:

```powershell
.\.venv\Scripts\Activate.ps1
```

PowerShell script çalıştırmaya izin vermiyorsa:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Daha sonra tekrar:

```powershell
.\.venv\Scripts\Activate.ps1
```

---

# 3. Python Bağımlılıklarını Kurma

Virtual environment aktifken:

```bash
pip install --upgrade pip
```

Ardından:

```bash
pip install -r requirements.txt
```

Kurulum tamamlandıktan sonra:

```bash
pip list
```

komutu ile paketleri kontrol edebilirsiniz.

---

# 4. Model Klasörlerini Oluşturma

Proje içerisinde model klasörlerini oluşturun.

Linux / macOS:

```bash
mkdir -p models/qwen
mkdir -p models/multilingual-e5-small
```

Windows PowerShell:

```powershell
mkdir models
mkdir models\qwen
mkdir models\multilingual-e5-small
```

Proje yapısı yaklaşık olarak şu şekilde olmalıdır:

```text
Files/
│
├── rag/
│   │
│   ├── backups/
│   ├── evaluation/
│   ├── incident_detection/
│   ├── log_agent/
│   ├── qdrant/
│   ├── scripts/
│   ├── training/
│   ├── training_management/
│   ├── training_pipeline/
│   ├── worker/
│   ├── export_training_dataset.py
│   └── migrate_telemetry_v2.py
│
├── tools/
│   └── whichllm
│
├── .gitignore
└── Readme.md

---

# 5. Qwen Modelini İndirme

Bu projede kullanılan model:

```text
Qwen3-4B-Instruct-2507
```

Quantized sürüm:

```text
Q4_K_M
```

Model formatı:

```text
GGUF
```

Önerilen model:

```text
Qwen3-4B-Instruct-2507.Q4_K_M.gguf
```

Modeli Hugging Face üzerinden indirebilirsiniz.

Örnek repository:

```text
MaziyarPanahi/Qwen3-4B-Instruct-2507-GGUF
```

İndirilen `.gguf` dosyasını şu klasöre koyun:

```text
models/qwen/
```

Son durumda:

```text
models/
└── qwen/
    └── Qwen3-4B-Instruct-2507.Q4_K_M.gguf
```

Model dosyasının gerçekten bulunduğunu kontrol edin.

Linux:

```bash
ls -lh models/qwen/
```

Windows:

```powershell
dir models\qwen
```

---

# 6. Hugging Face CLI ile Model İndirme

İsterseniz modeli terminal üzerinden de indirebilirsiniz.

Önce:

```bash
pip install -U huggingface_hub
```

Ardından:

```bash
huggingface-cli download \
MaziyarPanahi/Qwen3-4B-Instruct-2507-GGUF \
Qwen3-4B-Instruct-2507.Q4_K_M.gguf \
--local-dir models/qwen
```

Windows PowerShell:

```powershell
huggingface-cli download MaziyarPanahi/Qwen3-4B-Instruct-2507-GGUF Qwen3-4B-Instruct-2507.Q4_K_M.gguf --local-dir models\qwen
```

---

# 7. Embedding Modelini İndirme

RAG sistemi için kullanılan embedding modeli:

```text
intfloat/multilingual-e5-small
```

Terminal üzerinden indirebilirsiniz:

```bash
huggingface-cli download \
intfloat/multilingual-e5-small \
--local-dir models/multilingual-e5-small
```

Windows:

```powershell
huggingface-cli download intfloat/multilingual-e5-small --local-dir models\multilingual-e5-small
```

Sonrasında klasör yaklaşık olarak şöyle görünmelidir:

```text
models/
│
├── qwen/
│   └── Qwen3-4B-Instruct-2507.Q4_K_M.gguf
│
└── multilingual-e5-small/
    ├── config.json
    ├── tokenizer.json
    ├── tokenizer_config.json
    └── ...
```

---

# 8. Environment Dosyasını Oluşturma

Proje içerisinde:

```text
.env
```

dosyası oluşturun.

Örnek:

```env
QDRANT_URL=http://localhost:6333

MODEL_PATH=models/qwen/Qwen3-4B-Instruct-2507.Q4_K_M.gguf

EMBEDDING_MODEL_PATH=models/multilingual-e5-small

AI_SERVICE_KEY=change-this-key

HOST=0.0.0.0

PORT=8000
```

Production ortamında:

```env
AI_SERVICE_KEY=change-this-key
```

yerine güçlü ve rastgele bir API key kullanın.

Örnek:

```env
AI_SERVICE_KEY=9fa33e6913f34f8ca8b325aa
```

`.env` dosyasını GitHub'a yüklemeyin.

---

# 9. Qdrant'ı Çalıştırma

RAG sistemi için Qdrant gerekir.

Docker'ın çalıştığından emin olun.

Ardından:

```bash
docker run -d \
  --name qdrant \
  -p 6333:6333 \
  -p 6334:6334 \
  -v qdrant_storage:/qdrant/storage \
  qdrant/qdrant
```

Windows PowerShell tek satır:

```powershell
docker run -d --name qdrant -p 6333:6333 -p 6334:6334 -v qdrant_storage:/qdrant/storage qdrant/qdrant
```

Qdrant container durumunu kontrol edin:

```bash
docker ps
```

Listede:

```text
qdrant
```

görünmelidir.

---

# 10. Qdrant Bağlantısını Test Etme

Browser üzerinden:

```text
http://localhost:6333
```

veya terminalden:

```bash
curl http://localhost:6333
```

Windows PowerShell:

```powershell
Invoke-WebRequest http://localhost:6333
```

Qdrant cevap veriyorsa sistem hazırdır.

---

# 11. AI Servisini Çalıştırma

Virtual environment aktif olmalıdır.

Linux / macOS:

```bash
source .venv/bin/activate
```

Windows:

```powershell
.\.venv\Scripts\Activate.ps1
```

Ardından FastAPI uygulamasını başlatın.

Örnek:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Development için:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

> Eğer FastAPI giriş dosyanız farklıysa `app.main:app` kısmını kendi proje yapınıza göre değiştirin.

Örneğin uygulama:

```text
main.py
```

dosyasındaysa:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

---

# 12. Servisin Çalıştığını Kontrol Etme

Browser:

```text
http://localhost:8000
```

Health endpoint:

```text
http://localhost:8000/health
```

Terminal:

```bash
curl http://localhost:8000/health
```

Beklenen cevap örneği:

```json
{
  "status": "ok"
}
```

---

# 13. Swagger API Arayüzü

FastAPI Swagger arayüzüne:

```text
http://localhost:8000/docs
```

adresinden ulaşabilirsiniz.

Buradan API endpointlerini doğrudan test edebilirsiniz.

Alternatif dokümantasyon:

```text
http://localhost:8000/redoc
```

---

# 14. Modeli Kullanma

RAG tabanlı AI servisine soru göndermek için:

```text
POST /api/log-agent/chat
```

endpointini kullanın.

Örnek:

```bash
curl -X POST "http://localhost:8000/api/log-agent/chat" \
-H "Content-Type: application/json" \
-H "X-AI-Service-Key: YOUR_API_KEY" \
-d '{
  "message": "Son sistem loglarında kritik bir hata var mı?"
}'
```

`YOUR_API_KEY` yerine `.env` içerisinde belirlediğiniz:

```env
AI_SERVICE_KEY
```

değerini kullanın.

---

# 15. Python ile Model Kullanımı

Önce:

```bash
pip install requests
```

Daha sonra:

```python
import requests

url = "http://localhost:8000/api/log-agent/chat"

headers = {
    "Content-Type": "application/json",
    "X-AI-Service-Key": "YOUR_API_KEY"
}

payload = {
    "message": "Son sistem loglarında kritik bir hata var mı?"
}

response = requests.post(
    url,
    headers=headers,
    json=payload,
    timeout=120
)

print(response.status_code)
print(response.json())
```

---

# 16. Raw Qwen Kullanımı

RAG kullanmadan doğrudan Qwen modeline soru göndermek için proje destekliyorsa:

```text
POST /api/qwen/chat
```

kullanılabilir.

Örnek:

```bash
curl -X POST "http://localhost:8000/api/qwen/chat" \
-H "Content-Type: application/json" \
-H "X-AI-Service-Key: YOUR_API_KEY" \
-d '{
  "message": "Docker nedir?"
}'
```

---

# 17. RAG Modeli ile Raw Model Arasındaki Fark

## RAG

```text
/api/log-agent/chat
```

RAG modeli:

- Qdrant kullanır
- Sisteme ait bilgileri arar
- Logları analiz eder
- Dokümanlardan bilgi getirir
- Qwen modeline bulunan bilgileri context olarak verir

Örnek:

```text
Backend dün neden durmuş?
```

Bu tarz sorular için RAG kullanın.

---

## Raw Qwen

```text
/api/qwen/chat
```

Raw model:

- Qdrant kullanmaz
- Sisteme özel verilere erişmez
- Genel LLM cevabı üretir

Örnek:

```text
Docker container nedir?
```

---

# 18. Projeyi Tekrar Açmak

Bilgisayarı yeniden başlattıktan sonra her şeyi tekrar kurmanız gerekmez.

Sadece servisleri yeniden çalıştırmanız yeterlidir.

Önce Docker'ı açın.

Qdrant daha önce oluşturulduysa:

```bash
docker start qdrant
```

Ardından proje dizinine girin:

```bash
cd huggingface-model-server
```

Virtual environment'i açın.

Linux:

```bash
source .venv/bin/activate
```

Windows:

```powershell
.\.venv\Scripts\Activate.ps1
```

Son olarak AI servisini çalıştırın:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

---

# 19. Servisleri Kapatma

FastAPI çalışan terminalde:

```text
CTRL + C
```

Qdrant'ı durdurmak için:

```bash
docker stop qdrant
```

Tekrar başlatmak için:

```bash
docker start qdrant
```

---

# 20. Model Dosyası Bulunamadı Hatası

Önce:

```bash
ls models/qwen
```

Windows:

```powershell
dir models\qwen
```

Şu dosyanın bulunduğundan emin olun:

```text
Qwen3-4B-Instruct-2507.Q4_K_M.gguf
```

Daha sonra `.env` dosyasını kontrol edin:

```env
MODEL_PATH=models/qwen/Qwen3-4B-Instruct-2507.Q4_K_M.gguf
```

---

# 21. Qdrant Bağlantı Hatası

Docker container durumunu kontrol edin:

```bash
docker ps
```

Qdrant çalışmıyorsa:

```bash
docker start qdrant
```

Hâlâ çalışmıyorsa:

```bash
docker logs qdrant
```

---

# 22. Port 8000 Kullanılıyor Hatası

Linux:

```bash
sudo lsof -i :8000
```

Windows:

```powershell
netstat -ano | findstr :8000
```

Alternatif port kullanabilirsiniz:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8001
```

Bu durumda API adresiniz:

```text
http://localhost:8001
```

olur.

---

# 23. API 401 Unauthorized Hatası

Bu hata genellikle API key yanlış olduğunda oluşur.

`.env`:

```env
AI_SERVICE_KEY=my-secret-key
```

Request:

```text
X-AI-Service-Key: my-secret-key
```

İki değer aynı olmalıdır.

---

# 24. API 500 Internal Server Error

FastAPI terminalindeki logları kontrol edin.

Kontrol edilmesi gerekenler:

```text
MODEL_PATH
EMBEDDING_MODEL_PATH
QDRANT_URL
AI_SERVICE_KEY
```

Ayrıca Qdrant'ın çalıştığını kontrol edin:

```bash
docker ps
```

---

# 25. Model Çok Yavaş Çalışıyorsa

LLM'in çalışma hızı bilgisayar donanımına bağlıdır.

Performansı etkileyen başlıca faktörler:

- CPU
- GPU
- RAM
- VRAM
- Model quantization seviyesi
- Context uzunluğu
- Üretilen token sayısı

Bu projede kullanılan:

```text
Q4_K_M
```

quantization seviyesi, model boyutu ve kalite arasında dengeli bir seçenektir.

---

# 26. Minimum Sistem Gereksinimleri

Önerilen minimum:

```text
CPU: Modern 4+ core processor
RAM: 8 GB minimum
Disk: 10 GB boş alan
Python: 3.10+
Docker: Güncel sürüm
```

Daha rahat kullanım:

```text
RAM: 16 GB+
GPU: NVIDIA GPU
SSD: Önerilir
```

---

# 27. Model Dosyaları Neden GitHub'da Yok?

LLM ağırlıkları birkaç GB boyutunda olabilir.

Bu nedenle şu dosyalar GitHub repository içerisinde tutulmaz:

```text
models/
*.gguf
*.bin
*.safetensors
*.onnx
```

Repository yalnızca:

```text
source code
configuration
requirements
scripts
documentation
```

içerir.

Model dosyaları kurulum sırasında ayrıca indirilir.

---

# 28. GitHub'dan İndirdikten Sonra Hızlı Kurulum

Linux / macOS:

```bash
git clone YOUR_REPOSITORY_URL

cd huggingface-model-server

python3 -m venv .venv

source .venv/bin/activate

pip install -r requirements.txt

mkdir -p models/qwen

pip install huggingface_hub

huggingface-cli download \
MaziyarPanahi/Qwen3-4B-Instruct-2507-GGUF \
Qwen3-4B-Instruct-2507.Q4_K_M.gguf \
--local-dir models/qwen

huggingface-cli download \
intfloat/multilingual-e5-small \
--local-dir models/multilingual-e5-small

docker run -d \
--name qdrant \
-p 6333:6333 \
-p 6334:6334 \
-v qdrant_storage:/qdrant/storage \
qdrant/qdrant

uvicorn app.main:app --host 0.0.0.0 --port 8000
```

---

# 29. Windows Hızlı Kurulum

PowerShell:

```powershell
git clone YOUR_REPOSITORY_URL

cd huggingface-model-server

python -m venv .venv

.\.venv\Scripts\Activate.ps1

pip install -r requirements.txt

mkdir models
mkdir models\qwen
mkdir models\multilingual-e5-small

pip install huggingface_hub

huggingface-cli download MaziyarPanahi/Qwen3-4B-Instruct-2507-GGUF Qwen3-4B-Instruct-2507.Q4_K_M.gguf --local-dir models\qwen

huggingface-cli download intfloat/multilingual-e5-small --local-dir models\multilingual-e5-small

docker run -d --name qdrant -p 6333:6333 -p 6334:6334 -v qdrant_storage:/qdrant/storage qdrant/qdrant

uvicorn app.main:app --host 0.0.0.0 --port 8000
```

---

# 30. Kurulum Sonrası

Her şey başarıyla çalışıyorsa:

```text
Qdrant:
http://localhost:6333
```

```text
AI Service:
http://localhost:8000
```

```text
Swagger:
http://localhost:8000/docs
```

```text
Health:
http://localhost:8000/health
```

---

# Teknolojiler

- Python
- FastAPI
- Qwen
- GGUF
- Qdrant
- RAG
- multilingual-e5-small
- Docker
- REST API

---

# Kullanılan LLM

```text
Qwen3-4B-Instruct-2507
```

Quantization:

```text
Q4_K_M
```

Format:

```text
GGUF
```

---

# Önemli

Aşağıdaki dosyaları GitHub repository'ye yüklemeyin:

```text
.env
models/
.venv/
*.gguf
*.safetensors
*.bin
*.key
*.pem
qdrant_storage/
```

API key, şifre veya private server bilgilerini kaynak kod içerisinde paylaşmayın.