@echo off
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
cd /d E:\PROJECT_OF_THE_CLASS
"C:\Users\hp\AppData\Local\Programs\Python\Python311\python.exe" -m streamlit run app/streamlit_app.py --server.port 8501 --server.headless true --server.fileWatcherType none --browser.gatherUsageStats false > server.log 2>&1
