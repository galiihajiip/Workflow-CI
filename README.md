# Workflow-CI - Galih Aji Pangestu

Repository Kriteria 3 submission kelas Membangun Sistem Machine Learning (Dicoding).

## Struktur

```
Workflow-CI/
├── .github/workflows/ci.yml
└── MLProject/
    ├── MLProject
    ├── conda.yaml
    ├── modelling.py
    └── diabetes_preprocessing/
        ├── diabetes_clean.csv
        ├── train.csv
        └── test.csv
```

## CI Workflow

GitHub Actions akan menjalankan training model otomatis saat:
- Ada push ke branch `main` yang mengubah file di `MLProject/`
- Workflow dijalankan manual lewat `workflow_dispatch`

Workflow menjalankan `mlflow run . --env-manager=local` di folder `MLProject`.

## Cara Menjalankan Lokal

```bash
cd MLProject
mlflow run . --env-manager=local
```
