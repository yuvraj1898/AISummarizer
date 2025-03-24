# 📄 AI-Powered Summarizer (PDF, YouTube & Web URLs)
### App Link:https://aisummarizer-mznte74r5wosv7oawkktfk.streamlit.app/
This is a **Streamlit-based AI-powered summarizer** that can extract and summarize content from:
- **PDF files**
- **YouTube videos (Transcripts)**
- **Web pages (Static content scraping)**

The app allows users to:
✅ **Choose an LLM model dynamically from Groq API**  
✅ **Customize summary style (Concise, Detailed, Technical, Creative)**  
✅ **Adjust model creativity using a temperature slider**  
✅ **Rephrase summaries if not satisfied**  
✅ **Download summaries as `.txt` files**  

---

## 🚀 Features
- **📂 PDF Upload & Summarization**  
- **🎥 YouTube Transcript Summarization**  
- **🌍 Website Content Summarization**  
- **📜 Multiple Summary Styles**  
- **🛠️ Customizable Model Selection (Groq API)**  
- **♻️ Rephrase Summary Button**  
- **📥 Download Summary as `.txt`**

---

## 🛠️ Installation

### **1️⃣ Clone the Repository**
```bash
git clone https://github.com/yuvraj1898/AISummarizer.git
cd AI-Document-Summarizer
```
```bash
python -m venv venv
```
```bash
source venv/bin/activate  # On macOS/Linux
venv\Scripts\activate     # On Windows
```
```bash
pip install -r requirements.txt
```
## 🔑 API Key Setup
- **This project uses Groq API for model inference.**
- **Create a .env file in the project root and add:**
```bash
GROQ_API_KEY=your_api_key_here
Or enter the API key directly in the Streamlit sidebar.
```
## ▶️ Run the App
```bash
streamlit run app.py
Note: If the file is named differently, replace app.py with the correct filename.
```
## 📌 Usage
### 1️⃣ Select an LLM Model
Enter your Groq API key in the sidebar.

Select an LLM model from the dropdown.

### 2️⃣ Upload a PDF / Enter a YouTube or Web URL
For PDFs: Upload a file via the UI.

For YouTube: Paste a video URL (fetches transcript).

For Websites: Paste a webpage URL (extracts text content).

### 3️⃣ Customize Summary Preferences
Choose a summary style:

Concise

Detailed

Technical

Creative

Adjust temperature for creativity.

### 4️⃣ Generate & Rephrase Summary
Click Summarize to generate output.

Click ♻️ Rephrase Summary if not satisfied.

Download the summary using 📥 Download Summary as .txt.

Update Via API

