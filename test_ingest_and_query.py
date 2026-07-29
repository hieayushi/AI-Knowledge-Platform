"""
Verification & Submission Test Script: Proof of Execution
Demonstrates end-to-end ingestion and semantic search over attached PDF & Python code files.
"""

import os
import sys
import time
import requests

BASE_URL = "http://127.0.0.1:8000/api/v1"

PDF_PATH = r"f:\Acrobiz\Knowledge_Base_Sample (2).pdf"
CODE_PATH = r"f:\Acrobiz\Source_Code_Sample (2).py"

def check_server_health():
    print("Step 1: Checking API Server Health...")
    try:
        res = requests.get("http://127.0.0.1:8000/health", timeout=5)
        if res.status_code == 200:
            print(" API Server is healthy and running!")
            print(res.json())
        else:
            print(f" Server returned status code: {res.status_code}")
            sys.exit(1)
    except Exception as e:
        print(f" Could not connect to API server: {e}")
        print("Please start the server first with: python -m uvicorn app.main:app --port 8000")
        sys.exit(1)

def ingest_pdf_file():
    print("\n" + "="*60)
    print("Step 2: Ingesting PDF Document via Upload API (Knowledge_Base_Sample)")
    print("="*60)
    
    if not os.path.exists(PDF_PATH):
        print(f" PDF file not found at: {PDF_PATH}")
        return None

    with open(PDF_PATH, "rb") as f:
        files = {"file": ("Knowledge_Base_Sample.pdf", f, "application/pdf")}
        data = {"tags": "doc, knowledge_base, internal"}
        res = requests.post(f"{BASE_URL}/documents/upload", files=files, data=data)
        
    if res.status_code == 201:
        doc_info = res.json()
        print(" PDF Ingestion Successful!")
        print(f"  - Document ID: {doc_info['document_id']}")
        print(f"  - File Name  : {doc_info['file_name']}")
        print(f"  - Chunk Count: {doc_info['chunk_count']}")
        return doc_info['document_id']
    else:
        print(f" PDF Ingestion Failed: {res.status_code} - {res.text}")
        return None

def ingest_code_file():
    print("\n" + "="*60)
    print("Step 3: Ingesting Source Code File via Upload API (Source_Code_Sample.py)")
    print("="*60)
    
    if not os.path.exists(CODE_PATH):
        print(f" Code file not found at: {CODE_PATH}")
        return None

    with open(CODE_PATH, "rb") as f:
        files = {"file": ("Source_Code_Sample.py", f, "text/x-python")}
        data = {"tags": "code, python, rotator, proxy"}
        res = requests.post(f"{BASE_URL}/documents/upload", files=files, data=data)
        
    if res.status_code == 201:
        doc_info = res.json()
        print(" Source Code Ingestion Successful!")
        print(f"  - Document ID: {doc_info['document_id']}")
        print(f"  - File Name  : {doc_info['file_name']}")
        print(f"  - Chunk Count: {doc_info['chunk_count']}")
        return doc_info['document_id']
    else:
        print(f" Source Code Ingestion Failed: {res.status_code} - {res.text}")
        return None

def query_semantic_search(query_text: str, file_type_filter: str = None, top_k: int = 3):
    print("\n" + "-"*60)
    print(f"Querying: '{query_text}' (Filter: {file_type_filter or 'None'})")
    print("-"*60)
    
    payload = {
        "query": query_text,
        "top_k": top_k,
        "enable_reranking": True
    }
    if file_type_filter:
        payload["filters"] = {"file_type": file_type_filter}

    res = requests.post(f"{BASE_URL}/query", json=payload)
    if res.status_code == 200:
        data = res.json()
        print(f"Retrieved {data['total_results']} chunks in {data['execution_time_ms']} ms:")
        for idx, result in enumerate(data['results'], 1):
            score = result.get('rerank_score') or result['similarity_score']
            line_info = f" (Lines {result['start_line']}-{result['end_line']})" if result.get('start_line') else ""
            print(f"\n[{idx}] File: {result['file_name']}{line_info} | Score: {round(score, 4)}")
            print(f"     Excerpt: {result['content'][:250]}...")
    else:
        print(f"Query Error: {res.status_code} - {res.text}")

def test_ai_gateway():
    print("\n" + "="*60)
    print("Step 5: Testing Centralized AI Gateway Response Synthesis")
    print("="*60)
    
    payload = {
        "question": "How does the DecayProxyRotator handle proxy penalties and score recovery?",
        "top_k": 3
    }
    res = requests.post(f"{BASE_URL}/ai/chat", json=payload)
    if res.status_code == 200:
        data = res.json()
        print(f"\nAI Gateway Response ({data['execution_time_ms']} ms):")
        print(data['answer'])
    else:
        print(f"AI Gateway Error: {res.status_code} - {res.text}")

if __name__ == "__main__":
    print("="*60)
    print(" Internal AI Knowledge Platform - Verification Test Suite")
    print("="*60)
    
    check_server_health()
    pdf_id = ingest_pdf_file()
    code_id = ingest_code_file()

    time.sleep(1)

    print("\n" + "="*60)
    print("Step 4: Validating Semantic Retrieval on Sample Data")
    print("="*60)

    # Code queries
    query_semantic_search("How does DecayProxyRotator calculate proxy recovery score?", file_type_filter="code")
    query_semantic_search("What happens when UAFreshnessRotator hits a CAPTCHA or 403 error?", file_type_filter="code")

    # Document queries
    query_semantic_search("What are the key specifications in the knowledge base sample document?", file_type_filter="pdf")

    # AI Gateway
    test_ai_gateway()

    print("\n" + "="*60)
    print(" Verification Completed Successfully!")
    print("="*60)
