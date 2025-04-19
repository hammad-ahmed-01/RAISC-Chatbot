# 🚀 RAISC Chatbot

## 📌 Project Overview
RAISC Chatbot is an AI-powered chatbot system built using **FastAPI**. It stores chat history in **Firestore** and integrates **ChromaDB** for Retrieval-Augmented Generation (RAG). This is one of the 3 projects in the RAISC Tech Stack.

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

## 4️⃣ Run the Project
After setting up everything, start the FastAPI backend.
```sh
uvicorn main:app --reload --port 8000
```

### **Hopefully this works**

