#!/usr/bin/env python3
"""
Quick test script to verify Ollama integration.
Run this from the repo root: python tests/test_ollama.py
"""

from utils.ollama_client import ollama_chat, ollama_generate_json

def test_chat():
    print("=== Ollama Chat Test ===")
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Say 'Hello from Ollama!' in one sentence."},
    ]
    try:
        reply = ollama_chat(messages, temperature=0.0, max_tokens=100)
        print("Reply:", reply)
    except Exception as e:
        print("Error:", e)

def test_json():
    print("\n=== Ollama JSON Test ===")
    prompt = (
        "Return ONLY valid JSON with keys: subject, body, url.\n"
        "Subject: Welcome to our new product!\n"
        "Body: Try it now.\n"
        "URL: https://example.com\n"
    )
    try:
        parsed = ollama_generate_json(prompt, temperature=0.0, max_tokens=200)
        print("Parsed JSON:", parsed)
    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    test_chat()
    test_json()
