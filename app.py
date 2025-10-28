from flask import Flask, request, Response, stream_with_context
from flask_cors import CORS
import requests
import json
import os

app = Flask(__name__)
CORS(app)

# Get NVIDIA API key from environment variable
NVIDIA_API_KEY = os.environ.get('NVIDIA_API_KEY', '')
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"

@app.route('/v1/chat/completions', methods=['POST'])
def chat_completions():
    try:
        data = request.json
        
        # Extract parameters from OpenAI format
        messages = data.get('messages', [])
        model = data.get('model', 'meta/llama-3.1-405b-instruct')
        temperature = data.get('temperature', 0.7)
        max_tokens = data.get('max_tokens', 1024)
        stream = data.get('stream', False)
        
        # Prepare NVIDIA NIM request
        nim_payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream
        }
        
        headers = {
            "Authorization": f"Bearer {NVIDIA_API_KEY}",
            "Content-Type": "application/json"
        }
        
        if stream:
            # Handle streaming response
            nim_response = requests.post(
                f"{NVIDIA_BASE_URL}/chat/completions",
                json=nim_payload,
                headers=headers,
                stream=True
            )
            
            def generate():
                for line in nim_response.iter_lines():
                    if line:
                        yield line + b'\n'
            
            return Response(
                stream_with_context(generate()),
                content_type='text/event-stream'
            )
        else:
            # Handle non-streaming response
            nim_response = requests.post(
                f"{NVIDIA_BASE_URL}/chat/completions",
                json=nim_payload,
                headers=headers
            )
            
            return Response(
                nim_response.content,
                status=nim_response.status_code,
                content_type='application/json'
            )
    
    except Exception as e:
        return {
            "error": {
                "message": str(e),
                "type": "proxy_error"
            }
        }, 500

@app.route('/v1/models', methods=['GET'])
def list_models():
    """Return available models in OpenAI format"""
    try:
        headers = {
            "Authorization": f"Bearer {NVIDIA_API_KEY}",
            "Content-Type": "application/json"
        }
        
        nim_response = requests.get(
            f"{NVIDIA_BASE_URL}/models",
            headers=headers
        )
        
        return Response(
            nim_response.content,
            status=nim_response.status_code,
            content_type='application/json'
        )
    except Exception as e:
        return {
            "error": {
                "message": str(e),
                "type": "proxy_error"
            }
        }, 500

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return {"status": "healthy", "service": "NVIDIA NIM Proxy"}, 200

@app.route('/', methods=['GET'])
def home():
    """Home endpoint with usage instructions"""
    return {
        "message": "NVIDIA NIM to OpenAI API Proxy",
        "endpoints": {
            "/v1/chat/completions": "POST - Chat completions",
            "/v1/models": "GET - List available models",
            "/health": "GET - Health check"
        },
        "usage": "Set this URL as your OpenAI API base in Janitor AI",
        "deployed_on": "Render"
    }, 200

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)