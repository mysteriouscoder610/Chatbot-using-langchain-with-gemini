from flask import Flask, render_template, request, redirect, url_for
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv
import os

load_dotenv()

app = Flask(__name__)

# Initialize model
model = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

# Initialize chat history
chat_history = [
    SystemMessage(content='You are a helpful AI assistant.')
]

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        user_input = request.form["user_input"]
        chat_history.append(HumanMessage(content=user_input))

        if user_input.lower() == "exit":
            return redirect(url_for("index"))

        result = model.invoke(chat_history)
        chat_history.append(AIMessage(content=result.content))

    return render_template("index.html", messages=chat_history)

if __name__ == "__main__":
    app.run(debug=True)
