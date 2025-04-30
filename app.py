import os
import json
import threading
import atexit
from flask import Flask, render_template, jsonify, request
from chat_downloader import ChatDownloader
from nltk.sentiment import SentimentIntensityAnalyzer
from transformers import pipeline

app = Flask(__name__)

CHAT_FILE = "chat_messages.json"
OFFENSIVE_CHAT_FILE = "offensive_chat.json"

current_url = None
seen_messages = set()
chat_thread = None
stop_thread = threading.Event()

SENTIMENT_MODEL = "vader"

sia = SentimentIntensityAnalyzer()

sentiment_pipeline = pipeline("text-classification", model="tabularisai/multilingual-sentiment-analysis")


def analyze_sentiment(text):
    """setting the threshold and class for flagging"""
    print(SENTIMENT_MODEL)
    if SENTIMENT_MODEL == "vader":
        sentiment_score = sia.polarity_scores(text)["compound"]
        return sentiment_score < -0.2  # threshold for vader sntiment
    elif SENTIMENT_MODEL == "huggingface":
        result = sentiment_pipeline(text)[0]

        return "negative" in result["label"].lower()  # labels to flag for hugging face
    return False


@app.route("/")
def home():
    """ui for the tool"""
    return render_template("index.html")


@app.route("/json")
def json_data():
    """chat messages"""
    return jsonify(load_chat(CHAT_FILE))


@app.route("/offensive_json")
def offensive_json():
    """offensive messages"""
    return jsonify(load_chat(OFFENSIVE_CHAT_FILE))


@app.route("/set_sentiment_model", methods=["POST"])
def set_sentiment_model():
    """which model to use for sentiment analysis."""
    global SENTIMENT_MODEL
    data = request.json
    model = data.get("model")

    if model in ["vader", "huggingface"]:
        SENTIMENT_MODEL = model
        return jsonify(success=True, model=SENTIMENT_MODEL)
    return jsonify(success=False, message="Invalid model. Use 'vader' or 'huggingface'.")


@app.route("/get_sentiment_model", methods=["GET"])
def get_sentiment_model():
    """sentiment model in use"""
    return jsonify(model=SENTIMENT_MODEL)


@app.route("/start_chat", methods=["POST"])
def start_chat():
    """getting twtich chat messages"""
    global current_url, chat_thread, stop_thread

    data = request.json
    new_url = data.get("url")

    if new_url and new_url != current_url:
        print(f"Switching to new chat: {new_url}")
        current_url = new_url

        stop_thread.set()
        if chat_thread and chat_thread.is_alive():
            chat_thread.join()

        stop_thread.clear()
        reset_json_file()
        chat_thread = threading.Thread(target=write_chat, daemon=True)
        chat_thread.start()

        return jsonify(success=True)

    return jsonify(success=False)


@app.route("/reset", methods=["POST"])
def reset_connection():
    """Reset the chat connection state."""
    global current_url, chat_thread, stop_thread

    stop_thread.set()
    if chat_thread and chat_thread.is_alive():
        chat_thread.join(timeout=1.0)

    current_url = None
    stop_thread.clear()
    reset_json_file()

    return jsonify(success=True)


def write_chat():
    """Fetch Twitch chat, analyze messages, and save to appropriate files."""
    global seen_messages, stop_thread

    if not current_url:
        return

    chat = ChatDownloader().get_chat(current_url)

    with open(CHAT_FILE, "a", encoding="utf-8") as chat_file, \
            open(OFFENSIVE_CHAT_FILE, "a", encoding="utf-8") as offensive_file:

        try:
            for message in chat:
                if stop_thread.is_set():
                    print("Stopping chat downloader thread.")
                    break

                message_id = message.get("message_id")
                text = message.get("message", "")

                if message_id and message_id not in seen_messages:
                    seen_messages.add(message_id)

                    is_offensive = analyze_sentiment(text)
                    message["is_offensive"] = is_offensive  # Add offensive flag

                    json.dump(message, chat_file, ensure_ascii=False)
                    chat_file.write("\n")
                    chat_file.flush()

                    if is_offensive:
                        json.dump(message, offensive_file, ensure_ascii=False)
                        offensive_file.write("\n")
                        offensive_file.flush()
        except KeyboardInterrupt:
            print("Chat downloader stopped.")


def load_chat(filename):
    """displaying the last 50 messages in the ui"""
    if not os.path.exists(filename):
        return []
    with open(filename, "r", encoding="utf-8") as file:
        lines = file.readlines()
        messages = [json.loads(line) for line in lines]
    return messages[-50:] # can be adjusted here


def reset_json_file():
    """when new url is entered reset json"""
    global seen_messages
    seen_messages.clear()
    for file in [CHAT_FILE, OFFENSIVE_CHAT_FILE]:
        if os.path.exists(file):
            open(file, "w").close()
            print(f"File {file} reset.")


atexit.register(reset_json_file)

if __name__ == "__main__":
    app.run(debug=True)
