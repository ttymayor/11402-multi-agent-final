# 生成式人工智慧導論 - 期末實作

課程助教系統，這是一個以 Streamlit 建置的課程助教應用，支援教材摘要、課程問答、RAG 檢索與練習題生成。

## 如何運行

本專案使用 `uv` 作為 Python 環境與套件管理工具。若本機已有 `uv`，clone 專案後可執行：

```bash
uv sync
uv run streamlit run app.py
```

啟動後，請在 Streamlit 側邊欄輸入 Gemini API Key，即可上傳教材或貼上文字開始使用。

## 沒有 uv 的情況

目前尚未規劃複雜的專題目錄架構，主要應用程式集中在 `app.py`。若不使用 `uv`，也可以只下載 `app.py`，並自行安裝必要套件後執行：

```bash
pip install streamlit google-genai pypdf
streamlit run app.py
```

專案指定 Python `3.14`；若使用其他版本，請自行確認套件相容性。
