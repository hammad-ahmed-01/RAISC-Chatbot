# 🚀 RAISC Chatbot

## 📌 Project Overview
RAISC Chatbot is an AI-powered chatbot system built using **FastAPI**. It features:
- **Text Chat** with RAG (Retrieval-Augmented Generation) using ChromaDB
- **Voice Messages** using LiveKit STT for transcription 
- **Full Voice Calls** with LiveKit voice agents (Work in Progress)
- **Chat History** stored in Firestore
- **Mental Health Assistant** with sentiment analysis and session summaries

This is one of the 3 projects in the RAISC Tech Stack.

> **Note:** A Firebase project is already set up. Contact the **admin** for access to Firestore credentials.

---

## 🛠️ Setup Guide
This guide provides steps to install **Google Cloud SDK**, authenticate, set up **C++ Build Tools**, install dependencies, and run the project.

---

## 1️⃣ Install Google Cloud SDK
Google Cloud SDK is required for authentication and interacting with Firestore.

### 🔹 Step 1: Download & Install Google Cloud SDK
- Download the SDK from [Google Cloud SDK Installation](https://cloud.google.com/sdk/docs/install).
- Follow the installation instructions for **Windows**, **macOS**, or **Linux**.
- Restart your terminal after installation.

### 🔹 Step 2: Authenticate Google Cloud SDK

You would most probably be prompted to authenticate, so do that using the account you have connected to the firebase. Then choose the **raisc-20b3d** project. That would setup the default project.

Run the following in cmd to re authenticate as now it stores the json Credentials and previously it did'nt. 
```sh
gcloud auth application-default login
```
Make sure it says that Credentials saved.

---

## 2️⃣ Install Microsoft C++ Build Tools (Windows Only)
Some dependencies require C++ build tools to compile correctly.

### 🔹 Step 1: Install Microsoft Visual C++ Build Tools
- Download from [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/).
- Select the following during installation:
  - ✅ MSVC v142 - VS 2019 C++ x64/x86 Build Tools
  - ✅ Windows 10 or Windows 11 SDK
  - ✅ C++ CMake Tools for Windows
- Might take some time.

## 3️⃣ Install Python Dependencies
After setting up Google Cloud SDK and Build Tools, install the dependencies.

Run the following command to install required packages
```sh
pip install -r requirements.txt
```
If chromadb fails to install, then restart computer and run the following, as it might be due to Microsoft Visual C++ Build Tools not setting up.
```sh
pip install chromadb
```
If LiveKit agents fail to install:
```sh
pip install livekit-agents livekit-plugins-deepgram livekit-plugins-groq livekit-plugins-silero
```


## 4️⃣ Run the Project
After setting up everything, start the FastAPI backend.
```sh
uvicorn main:app --reload --port 8000
```


## 3️⃣ Environment Variables Setup
Create a `.env` file in the root directory with the following variables:

```env
# FastAPI Configuration
FASTAPI_BASE_URL=http://localhost:8000

# Groq API (for LLM)
GROQ_API_KEY=your_groq_api_key_here

# Google Cloud Firestore (handled by gcloud auth)
PROJECT_ID=raisc-20b3d
COLLECTION_NAME=chat_history

# LiveKit (for voice functionality)
LIVEKIT_URL=wss://your-livekit-url
LIVEKIT_API_KEY=your_livekit_api_key
LIVEKIT_API_SECRET=your_livekit_api_secret

# STT Services (choose one)
DEEPGRAM_API_KEY=your_deepgram_api_key
If using Whisper-turbo then:
GROQ_API_KEY=your_groq_api_key

# Django Backend (if using user management)
DJANGO_BACKEND_URL=https://web-production-deb22.up.railway.app

# Django Backend (For testing)
DJANGO_BACKEND_URL=Your_local_backend_server (http://127.0.0.1:8000/)

# For voice calls
AZURE_SPEECH_KEY=your_azure_speech_key (for Text-to-Speech)
AZURE_SPEECH_HOST=your_azure_speech_host
AZURE_SPEECH_REGION=your_azure_speech_region

GROQ_API_KEY=your_groq_api_key (For Speech-to-text and the LLM)

LIVEKIT_URL=wss://your-livekit-url
LIVEKIT_API_KEY=your_livekit_api_key
LIVEKIT_API_SECRET=your_livekit_api_secret


```

---
## 5️⃣ Run the Project

### 🔹 Method 1: FastAPI Only (Text Chat)
For text-only functionality:
```sh
uvicorn main:app --port 8001 
```
Note for developers: Please keep the port of backend and chatbot seperate.
### 🔹 Method 2: Full System (Text + Voice)
For complete functionality with voice messages and voice calls:

**Terminal 1: FastAPI Backend (for chat)**
```sh
uvicorn main:app --port 8001
```

**Terminal 2: STT Agent (for voice messages)**
```sh
python run_stt_agent.py download-files
python run_stt_agent.py dev
```

**Terminal 3: Voice Agent (for full voice calls)**
```sh
python voice_agent.py download-files
python voice_agent.py dev

If you want to talk to the agent in console then:
python voice_agent.py console

Note: Voice-Agent is a Work in progress, it may not work in parallel with STT agent for voice messages, it does work standalone
```

**Terminal 4: Frontend (if using Next.js)**
```sh
npm run dev
```
**Terminal 5: Django Backend**
```sh
python manage.py runserver

Note: A patient must be registered in backend database.
```


## 🎯 Features

### 💬 Text Chat
- Mental health conversation with RAG-enhanced responses
- Session management and history
- Sentiment analysis and risk detection
- Automated session summaries

### 🎙️ Voice Messages
- Click-to-record voice messages in text chat
- Real-time transcription using Deepgram/OpenAI Whisper
- Audio level indicators
- Seamless integration with text conversation

### 📞 Full Voice Calls (Work in Progress)
- Complete voice conversation with AI agent
- Real-time speech-to-text and text-to-speech
- Voice activity detection
- Natural conversation flow

### 📊 Analytics
- User sentiment tracking
- Session emotional analysis
- Doctor summary integration
- Past conversation context

---

## 🏗️ Architecture

```
Frontend (Next.js)
├── Text Chat Interface
├── Voice Message Recording
└── LiveKit Voice Calls

Backend (FastAPI)
├── Chat API endpoints
├── History management
├── User data services
└── RAG processing

Voice Services
├── STT Agent (voice → text)
├── Voice Agent (full voice calls)
└── LiveKit infrastructure

Storage
├── Firestore (chat history)
├── ChromaDB (RAG knowledge)
└── Django Backend (user profiles)
```

---

## 🔧 Troubleshooting

### Common Issues:
1. **ChromaDB installation fails**: Restart computer after installing C++ Build Tools
2. **Voice not working**: Check LiveKit environment variables and microphone permissions
3. **No transcription**: Verify STT API keys (Deepgram, Groq for whisper-turbo model)
4. **Firebase connection**: Ensure `gcloud auth application-default login` shows "Credentials saved"

### Debug Voice Issues:
- Check browser console for audio track detection
- Verify microphone permissions in browser
- Ensure STT agent is running with `python run_stt_agent.py dev`
- Check LiveKit connection status in frontend

---

## 🚀 Deployment Notes 

For production deployment:
1. Set up LiveKit server or use LiveKit Cloud
2. Configure proper CORS settings
3. Use environment-specific API keys
4. Set up proper SSL certificates for voice functionality
5. Configure Firestore security rules

---

**Hopefully this works!**
---