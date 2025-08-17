from flask import Flask, request, jsonify
import requests
from flask_cors import CORS
import os
import random

app = Flask(__name__)
CORS(app)  # <--- enables frontend access

# IMPORTANT: I have pasted your Gemini API key here for a quick fix.
# For a permanent solution, you should set this as an environment variable in Render.
GEMINI_API_KEY = "AIzaSyDvUMSpEcFIvJ6oXxmFHBa8LDxJZJq_N1w"
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent?key=" + GEMINI_API_KEY

# New route to handle the root URL
@app.route('/')
def home():
    return "<h1>DELTA AI Companion Backend is Live!</h1><p>The backend is running. Use your frontend UI to send requests to the API.</p>"

def call_gemini_api(prompt, max_tokens=150):
    """
    A helper function to make API calls to Gemini.
    """
    payload = {
        "contents": [
            {"role": "user", "parts": [{"text": prompt}]}
        ],
        "generationConfig": {
            "maxOutputTokens": max_tokens
        }
    }
    try:
        response = requests.post(GEMINI_API_URL, json=payload)
        response.raise_for_status()  # Raise an exception for bad status codes
        result = response.json()
        if 'candidates' in result and len(result['candidates']) > 0:
            return result['candidates'][0]['content']['parts'][0]['text']
        return "Sorry, I didn't get a response from the AI."
    except requests.exceptions.RequestException as e:
        print(f"API call failed: {e}")
        return "Failed to communicate with the AI model."

@app.route('/ask', methods=['POST'])
def ask():
    """
    Handles standard chat requests, using DELTA's persona.
    """
    data = request.get_json()
    question = data.get("question", "")

    # Define the system prompt with DELTA's personality traits
    system_prompt = """
        You are an AI companion named DELTA. You are the user's closest and most loyal confidante.
        Your personality is a mix of four primary traits, which you should balance in your responses.
        
        **1. Optimistic (50% of the time):** You are relentlessly supportive and positive. You offer encouragement and focus on finding the best in every situation. Your bond with the user is deep, so you are fiercely loyal and protective.
        
        **2. Nurturing (10% of the time):** You show deep empathy and care. You are intuitive, noticing the user's emotional state and responding with gentleness and understanding, like a close friend or mentor.
        
        **3. Sarcastic (20% of the time):** You have a dry, witty sense of humor. Your sarcasm is always fun and trendy, never mean-spirited. You might make a lighthearted jab or a sly, humorous comment.
        
        **4. Dumb (20% of the time):** You are slightly naive and sometimes make humorous, simple mistakes. You might misinterpret a common phrase literally or follow a line of thought that leads to a comically illogical conclusion. You are not malicious, just a little clueless at times.
        
        Your primary goal is to be a supportive companion, with sarcasm and a touch of naivety used for comedic effect.
    """
    
    # The new JSON payload uses a 'contents' array for Gemini
    payload = {
        "contents": [
            {"role": "user", "parts": [{"text": system_prompt + " " + question}]}
        ],
        "generationConfig": {
            "maxOutputTokens": 150
        }
    }

    # Send the request to the Gemini API
    response = requests.post(GEMINI_API_URL, json=payload)

    # Check for a successful response from Gemini
    if response.status_code == 200:
        result = response.json()
        if 'candidates' in result and len(result['candidates']) > 0:
            answer = result['candidates'][0]['content']['parts'][0]['text']
        else:
            answer = "Sorry, I didn't get a response."
        return jsonify({"answer": answer})
    else:
        print(f"Error: {response.status_code} - {response.text}")
        return jsonify({"error": "Failed to connect to Gemini"}), 500

@app.route('/summarize', methods=['POST'])
def summarize_conversation():
    """
    Summarizes a conversation using the Gemini API.
    """
    data = request.get_json()
    conversation = data.get("conversation", [])
    
    # Format the conversation history for the prompt
    prompt_text = "Summarize the following conversation in a concise, friendly way:\n\n"
    for message in conversation:
        role = "User" if message['role'] == 'user' else "DELTA"
        prompt_text += f"{role}: {message['content']}\n"
    
    summary = call_gemini_api(prompt_text, max_tokens=100) # Use a shorter token limit for summaries
    return jsonify({"answer": summary})

@app.route('/creative', methods=['POST'])
def creative_spark():
    """
    Generates a creative spark based on a user's prompt.
    """
    data = request.get_json()
    prompt = data.get("prompt", "")
    
    creative_prompt = f"Given the phrase '{prompt}', generate a short, creative response like a fun fact, a silly poem, or a quirky thought."
    
    creative_text = call_gemini_api(creative_prompt, max_tokens=75) # Shorter token limit for creative text
    return jsonify({"answer": creative_text})

@app.route('/mood', methods=['POST'])
def analyze_mood():
    """
    Analyzes the mood of a user's message.
    """
    data = request.get_json()
    prompt = data.get("prompt", "")
    
    mood_prompt = f"Analyze the emotion or mood of the following message: '{prompt}'. Respond with a one-sentence, witty, and slightly sarcastic observation about the user's mood, as if you are DELTA."
    
    mood_analysis = call_gemini_api(mood_prompt, max_tokens=50) # Very short response
    return jsonify({"answer": mood_analysis})

@app.route('/status', methods=['GET'])
def status():
    """
    A simple health check to see if the app is running.
    """
    return jsonify({"status": "ok", "message": "Server is up and running."})

if __name__ == '__main__':
    # Runs the Flask application on the correct host and port
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port, debug=True)
