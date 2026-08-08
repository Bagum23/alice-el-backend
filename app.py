import os
import time
from flask import Flask, request, jsonify
from openai import OpenAI

app = Flask(__name__)

client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
    timeout=3.8
)

# Память по сессиям Яндекс Диалогов.
# Для тестов этого достаточно.
sessions = {}

# Храним последние 6 сообщений:
# 3 реплики пользователя + 3 ответа Эла.
MAX_HISTORY_MESSAGES = 6


@app.get("/")
def health():
    return jsonify({
        "status": "ok",
        "service": "alice-el-backend"
    })


def build_messages(history, text):
    messages = [
        {
            "role": "system",
            "content": (
                "Ты голосовой помощник Эл. "
                "Отвечай как умный разговорный ассистент на русском языке. "
                "Всегда учитывай предыдущий контекст разговора. "
                "Если пользователь говорит 'он', 'она', 'у него', 'у нее', "
                "'это', 'там', 'тот', 'эта' и подобные слова, "
                "определи, к чему они относятся из предыдущих реплик. "
                "Не задавай уточняющий вопрос, если смысл можно понять "
                "из контекста. "
                "Отвечай коротко и естественно: обычно 1-3 предложения. "
                "Для простых фактических вопросов давай прямой ответ сразу. "
                "Не используй markdown, таблицы и длинные списки."
            )
        }
    ]

    messages.extend(history)

    messages.append({
        "role": "user",
        "content": text
    })

    return messages


@app.post("/ask")
def ask():
    started = time.time()

    try:
        data = request.get_json(silent=True) or {}

        is_alice = (
            "request" in data
            and "session" in data
        )

        if is_alice:
            req = data.get("request", {})
            session = data.get("session", {})

            session_id = session.get(
                "session_id",
                "default"
            )

            text = (
                req.get("command")
                or req.get("original_utterance")
                or ""
            ).strip()

        else:
            session_id = "manual-test"

            text = data.get(
                "text",
                ""
            ).strip()

        print(
            f"SESSION: {session_id} | INPUT: {text}",
            flush=True
        )

        if not text:
            answer = "Привет! Чем могу помочь?"

        else:
            history = sessions.get(
                session_id,
                []
            )

            messages = build_messages(
                history,
                text
            )

            openai_started = time.time()

            response = client.responses.create(
                model="gpt-5-mini",
                input=messages,
                max_output_tokens=300
            )

            openai_time = (
                time.time()
                - openai_started
            )

            print(
                f"OPENAI TIME: {openai_time:.2f}s",
                flush=True
            )

            answer = (
                response.output_text
                or ""
            ).strip()

            if not answer:
                print(
                    "EMPTY OPENAI OUTPUT",
                    flush=True
                )

                answer = (
                    "Не удалось быстро сформировать ответ. "
                    "Попробуйте повторить вопрос."
                )

            # Запоминаем только успешный обмен.
            history.append({
                "role": "user",
                "content": text
            })

            history.append({
                "role": "assistant",
                "content": answer
            })

            sessions[session_id] = (
                history[-MAX_HISTORY_MESSAGES:]
            )

        # Алисе длинный текст не нужен.
        answer = answer[:800]

        total_time = (
            time.time()
            - started
        )

        print(
            f"TOTAL TIME: {total_time:.2f}s",
            flush=True
        )

        if is_alice:
            return jsonify({
                "response": {
                    "text": answer,
                    "tts": answer,
                    "end_session": False
                },
                "session": data.get(
                    "session",
                    {}
                ),
                "version": data.get(
                    "version",
                    "1.0"
                )
            })

        return jsonify({
            "answer": answer
        })

    except Exception as e:
        total_time = (
            time.time()
            - started
        )

        print(
            f"ERROR after {total_time:.2f}s: {repr(e)}",
            flush=True
        )

        data = (
            request.get_json(
                silent=True
            )
            or {}
        )

        is_alice = (
            "request" in data
            and "session" in data
        )

        if is_alice:
            fallback = (
                "Ответ занял слишком много времени. "
                "Повторите вопрос."
            )

            return jsonify({
                "response": {
                    "text": fallback,
                    "tts": fallback,
                    "end_session": False
                },
                "session": data.get(
                    "session",
                    {}
                ),
                "version": data.get(
                    "version",
                    "1.0"
                )
            })

        return jsonify({
            "error": str(e)
        }), 500


if __name__ == "__main__":
    port = int(
        os.environ.get(
            "PORT",
            10000
        )
    )

    app.run(
        host="0.0.0.0",
        port=port
    )
