import os
from flask import Flask, request, jsonify
from openai import OpenAI

app = Flask(__name__)

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


@app.get("/")
def health():
    return jsonify({
        "status": "ok",
        "service": "alice-el-backend"
    })


@app.post("/ask")
def ask():
    try:
        data = request.get_json(silent=True) or {}

        # Запрос из Яндекс Диалогов
        is_alice = "request" in data and "session" in data

        if is_alice:
            alice_request = data.get("request", {})
            text = (
                alice_request.get("command")
                or alice_request.get("original_utterance")
                or ""
            ).strip()
        else:
            # Наш старый тестовый формат {"text": "..."}
            text = data.get("text", "").strip()

        if not text:
            if is_alice:
                answer = "Я готов. Задайте мне вопрос."
            else:
                return jsonify({"error": "No text provided"}), 400
        else:
            response = client.responses.create(
                model="gpt-5-mini",
                input=text
            )
            answer = response.output_text.strip()

        # Формат ответа для Алисы
        if is_alice:
            # Алиса принимает максимум 1024 символа в response.text
            answer = answer[:1024]

            return jsonify({
                "response": {
                    "text": answer,
                    "tts": answer,
                    "end_session": False
                },
                "session": data.get("session", {}),
                "version": data.get("version", "1.0")
            })

        # Старый формат для PowerShell-теста
        return jsonify({
            "answer": answer
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
