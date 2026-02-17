# Official Fish-Speech image use karo (latest version, GPU ready)
FROM fishaudio/fish-speech:latest

# Apna custom handler file copy karo
WORKDIR /app
COPY handler.py .

# RunPod serverless ke liye yeh command chalegi (handler.py ko run karega)
CMD ["python3", "-u", "handler.py"]