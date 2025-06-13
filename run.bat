@echo off
set STREAMLIT_WATCHER_TYPE=none
uv venv .venv
call .venv\Scripts\activate
streamlit run main.py
