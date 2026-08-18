# Qwen IoT Lab Assistant — Model Usage Guide

Bu doküman, **Qwen IoT Lab Assistant** modelinin kurulumu, çalıştırılması ve API üzerinden kullanılması için hazırlanmıştır.

## 1. Proje Hakkında

Bu proje, kurum/laboratuvar verileri üzerinde çalışan yerel bir LLM ve RAG tabanlı yapay zekâ asistanıdır.

Temel bileşenler:

- **Qwen3-4B-Instruct-2507 Q4_K_M**
- **FastAPI**
- **Qdrant**
- **multilingual-e5-small**
- Hibrit RAG / vektör arama
- Türkçe ve İngilizce soru-cevap desteği
- REST API üzerinden kullanım

Ana model dosyası Git deposunda tutulmaz. Model ağırlıkları ayrıca indirilmelidir.

---

## 2. Proje Dizini

Örnek proje yapısı:

```text
huggingface-model-server/
├── app/
├── scripts/
├── rag/
│   ├── log_agent/
│   └── worker/
├── models/
│   ├── qwen/
│   └── multilingual-e5-small/
├── requirements.txt
├── Dockerfile
├── .env
└── MODEL_USAGE.md