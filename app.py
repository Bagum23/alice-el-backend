import os
import time
from flask import Flask, request, jsonify
from openai import OpenAI

app = Flask(__name__)

client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
    timeout=3.2
)


@app.get("/")
def health():
    return jsonify({
        "status": "ok",
        "service": "alice-el-backend"
    })


@app.post("/ask")
def ask():
    started = time.time()

    try:
        data = request.get_json(silent=True) or {}

        is_alice = "request" in data and "session" in data

        if is_alice:
            req = data.get("request", {})

            text = (
                req.get("command")
                or req.get("original_utterance")
                or ""
            ).strip()

        else:
            text = data.get("text", "").strip()

        print(f"INPUT: {text}", flush=True)

        if not text:
            answer = "Привет! Чем могу помочь?"
        else:
            openai_started = time.time()

response = client.responses.create(
    model="gpt-5.6-terra",
    instructions=(
        "Ты голосовой помощник Эл. "
        "Отвечай по-русски очень кратко и по существу. "
        "Один-два коротких предложения. "
        "Не более 35 слов. "
        "Не используй списки, markdown и длинные вступления."
    ),
    input=text,
    reasoning={
        "effort": "low"
    },
    max_output_tokens=120
)

            answer = response.output_text.strip()

            print(
                f"OPENAI TIME: {time.time() - openai_started:.2f}s",
                flush=True
            )

        if not answer:
            answer = "Не удалось сформировать ответ."

        # Для голосового интерфейса длинные ответы не нужны
        answer = answer[:900]

        total_time = time.time() - started
        print(f"TOTAL TIME: {total_time:.2f}s", flush=True)

        if is_alice:
            return jsonify({
                "response": {
                    "text": answer,
                    "tts": answer,
                    "end_session": False
                },
                "session": data.get("session", {}),
                "version": data.get("version", "1.0")
            })

        return jsonify({
            "answer": answer
        })

    except Exception as e:
        total_time = time.time() - started

        print(
            f"ERROR after {total_time:.2f}s: {repr(e)}",
            flush=True
        )

        # Алисе лучше вернуть корректный ответ,
        # чем HTTP 500
        data = request.get_json(silent=True) or {}
        is_alice = "request" in data and "session" in data

        if is_alice:
            fallback = (
                "Ответ занял слишком много времени. "
                "Попробуйте задать вопрос ещё раз."
            )

            return jsonify({
                "response": {
                    "text": fallback,
                    "tts": fallback,
                    "end_session": False
                },
                "session": data.get("session", {}),
                "version": data.get("version", "1.0")
            })

        return jsonify({
            "error": str(e)
        }), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
