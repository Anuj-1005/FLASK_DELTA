# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import requests

app = Flask(__name__)
CORS(app)  # allow calls from your Vercel/any frontend

# -------- Config --------
# Point this to your Ollama server.
# Local dev:  http://localhost:11434
# If deploying Flask on Render, OLLAMA_API_URL must be a public URL to an Ollama you control.
OLLAMA_API_URL = os.getenv("OLLAMA_API_URL", "http://localhost:11434")
MODEL_NAME = os.getenv("MODEL", "llama3")
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT_SECONDS", "60"))

# -------- Helpers --------
def call_ollama_chat(messages, model=MODEL_NAME, num_predict=256):
    """
    Calls Ollama's /api/chat endpoint with the given messages.
    messages: list of {"role": "system"|"user"|"assistant", "content": "text"}
    Returns a string (assistant content).
    """
    url = f"{OLLAMA_API_URL.rstrip('/')}/api/chat"
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {"num_predict": num_predict},
    }
    try:
        resp = requests.post(url, json=payload, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        # Expected shape: { "message": {"role": "assistant", "content": "..."} }
        if isinstance(data, dict):
            msg = data.get("message", {})
            content = msg.get("content")
            if content:
                return content
        # Fallback (some versions return an array of messages or different keys)
        # Try to stitch together any 'content' strings we can find:
        if isinstance(data, dict) and "messages" in data and isinstance(data["messages"], list):
            parts = [m.get("content", "") for m in data["messages"] if isinstance(m, dict)]
            if any(parts):
                return "\n".join(parts).strip()
        # Last resort
        return "I couldn't parse a reply from the model."
    except requests.exceptions.RequestException as e:
        print(f"[Ollama Error] {e}")
        return "❌ Failed to reach the AI model."
    except Exception as e:
        print(f"[Unexpected Parse Error] {e}")
        return "❌ Unexpected error parsing AI response."


# -------- Routes --------
@app.route("/", methods=["GET"])
def home():
    return (
        "<h1>DELTA (Llama3) Backend</h1>"
        "<p>Status: live. Use <code>/chat</code> (POST) to talk to the model.</p>"
    )

@app.route("/status", methods=["GET"])
def status():
    return jsonify({
        "status": "ok",
        "model": MODEL_NAME,
        "ollama_api_url": OLLAMA_API_URL
    })

@app.route("/chat", methods=["POST"])
def chat():
    """
    Body accepts either:
      { "question": "..." }  or  { "message": "..." }
    Responds with:
      { "answer": "..." }
    """
    data = request.get_json(silent=True) or {}
    user_text = data.get("question") or data.get("message") or ""

    if not user_text.strip():
        return jsonify({"answer": "⚠️ Please provide a question/message."}), 400

    # You can tweak the system prompt / persona safely here.
    system_prompt = (
        "You are DELTA, a friendly, upbeat AI companion. Be helpful, concise, and supportive. "
        "Use light humor when appropriate, but avoid offensive content. Keep responses clear."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_text},
    ]

    answer = call_ollama_chat(messages, model=MODEL_NAME, num_predict=256)
    return jsonify({"answer": answer})

@app.route("/summarize", methods=["POST"])
def summarize():
    data = request.get_json(silent=True) or {}
    conversation = data.get("conversation", [])
    if not isinstance(conversation, list) or not conversation:
        return jsonify({"answer": "⚠️ Provide a 'conversation' list."}), 400

    # Build a short summary prompt
    convo_text = []
    for m in conversation:
        role = m.get("role", "user")
        content = m.get("content", "")
        convo_text.append(f"{role.title()}: {content}")
    summary_prompt = (
        "Summarize the following conversation in 3-5 concise bullet points, focusing on key decisions and questions:\n\n"
        + "\n".join(convo_text)
    )

    messages = [
        {"role": "system", "content": "You are an expert summarizer."},
        {"role": "user", "content": summary_prompt},
    ]
    answer = call_ollama_chat(messages, model=MODEL_NAME, num_predict=180)
    return jsonify({"answer": answer})

@app.route("/creative", methods=["POST"])
def creative():
    data = request.get_json(silent=True) or {}
    prompt = data.get("prompt", "").strip()
    if not prompt:
        return jsonify({"answer": "⚠️ Provide a 'prompt'."}), 400

    creative_prompt = (
        f"Using the phrase: '{prompt}', create a short, clever 2-3 line piece (fun fact, poem, or witty thought)."
    )
    messages = [
        {"role": "system", "content": "You are a creative writing assistant."},
        {"role": "user", "content": creative_prompt},
    ]
    answer = call_ollama_chat(messages, model=MODEL_NAME, num_predict=120)
    return jsonify({"answer": answer})

@app.route("/mood", methods=["POST"])
def mood():
    data = request.get_json(silent=True) or {}
    text = data.get("prompt", "").strip()
    if not text:
        return jsonify({"answer": "⚠️ Provide a 'prompt'."}), 400

    mood_prompt = (
        f"Analyze the mood of this message in one crisp sentence with friendly, light humor:\n\n{text}"
    )
    messages = [
        {"role": "system", "content": "You analyze tone and sentiment."},
        {"role": "user", "content": mood_prompt},
    ]
    answer = call_ollama_chat(messages, model=MODEL_NAME, num_predict=80)
    return jsonify({"answer": answer})

# -------- Entry --------
if __name__ == "__main__":
    # Render provides PORT; default for local dev
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=True)
