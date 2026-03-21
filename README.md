# WorkflowCI_GalihAjiPangestu

Repository untuk CI Workflow - Kriteria 3 Submission SMSML

**Nama**: Galih Aji Pangestu  
**Username Dicoding**: gapzzzzzzzzz

## Struktur
```
MLProject/
├── MLProject          # MLflow project config
├── conda.yaml         # Environment dependencies
├── modelling.py       # Training script
├── dockerhub.txt      # Docker Hub image link
└── insurance_preprocessing/  # Preprocessed dataset
```

## CI Workflow
Workflow otomatis berjalan saat push ke `main`:
1. Setup Python 3.12.7
2. Install MLflow
3. Run `mlflow run MLProject`
4. Commit & push mlruns artifact ke repo
5. Build Docker image dengan `mlflow models build-docker`
6. Push image ke Docker Hub
