from flask import Flask, request, jsonify
import requests
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # <--- enables frontend access

@app.route('/ask', methods=['POST'])
def ask():
    """
    Handles POST requests to the /ask endpoint.
    It takes a user's question and sends it to the Ollama API with a
    system prompt to define DELTA's personality and a token limit.
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

    # The new JSON payload uses a 'messages' array and 'options'
    payload = {
        "model": "llama3",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question}
        ],
        "options": {
            # This is the key change to limit the response length
            # Setting max 150 tokens as requested
            "num_predict": 150,
            # You can also add other options like temperature here
        },
        "stream": False
    }

    # Send the request to the Ollama API
    response = requests.post("http://localhost:11434/api/chat", json=payload)

    # Check for a successful response from Ollama
    if response.status_code == 200:
        result = response.json()
        answer = result.get("message", {}).get("content", "Sorry, I didn't get a response.")
        return jsonify({"answer": answer})
    else:
        # If the request failed, return an error message
        print(f"Error: {response.status_code} - {response.text}")
        return jsonify({"error": "Failed to contact Ollama"}), 500

if __name__ == '__main__':
    # Runs the Flask application in debug mode
    app.run(debug=True)