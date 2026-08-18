# Log Agent Candidate Exporter v2

## Sunucuya kopyalama

```powershell
scp `
  "$HOME\Downloads\log-agent-benchmark-exporter-v2.zip" `
  dell@10.142.1.136:/home/dell/
```

## ZIP'i Python ile çıkarma

```bash
cd /home/dell

python3 - <<'PY'
from pathlib import Path
from zipfile import ZipFile

source = Path("/home/dell/log-agent-benchmark-exporter-v2.zip")
target = Path(
    "/home/dell/huggingface-model-server/"
    "rag/scripts/log_agent_benchmark"
)
target.mkdir(parents=True, exist_ok=True)

with ZipFile(source) as archive:
    archive.extractall(target)

print("Çıkarıldı:", target)
PY
```

## Çalıştırma

```bash
cd ~/huggingface-model-server

.rag-venv/bin/python \
  rag/scripts/log_agent_benchmark/export_candidates_v2.py \
  --candidate-count 80 \
  --max-per-category 15
```

## Özet

```bash
.rag-venv/bin/python -m json.tool \
  rag/data/log_agent_benchmark/candidates_v2.summary.json
```
