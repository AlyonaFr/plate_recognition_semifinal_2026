# Публикация проекта в GitHub/GitFlic

## Перед коммитом

```powershell
pip install "scikit-learn==1.8.0" --force-reinstall
py main.py --input test_images --output results\result.csv --benchmark
py -m pytest -q
py dataset\validate_dataset.py --root dataset
```

## Git

```powershell
git init
git branch -M main
git add .
git status
git commit -m "Semifinal license plate recognition solution"
```

После создания пустого репозитория на GitHub/GitFlic:

```powershell
git remote add origin https://github.com/USERNAME/plate-recognition-semifinal.git
git push -u origin main
```

Не добавляйте `.venv/` в репозиторий.
