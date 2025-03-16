import os
from pathlib import Path

list_of_files = [
	"QAWithPDF/__init__.py",
	"QAWithPDF/data_ingestion.py",
	"QAWithPDF/embedding.py",
	"QAWithPDF/model_api.py",
	"QAWithPDF/app.py",
	"logger.py",
	"exception.py",
	"setup.py",
]

for filepath in list_of_files:
	filepath = Path(filepath)
	filedir, filename = os.path.split(filepath)
	if filedir != "":
		os.makedirs(filedir, exist_ok=True)
		print(f"Creating directory: {filedir} for file: {filename}")
	if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
		with open(filepath, "w") as f:
			pass
			print(f"Creating empty file: {filepath}")
	else:
		print(f"{filename} already exists")