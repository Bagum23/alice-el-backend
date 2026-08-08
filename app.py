import os
import time
from flask import Flask, request, jsonify
from openai import OpenAI

app = Flask(__name__)

client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
    timeout=4.0
)

# Простая память диалога в RAM.
# Ключ = session_id Яндекс Диалогов
# Значение = список последних сообщений
sessions = {}

MAX_HISTORY_MESSAGES = 8


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

        is_alice = (
            "request" in data
            and "session" in data
        )

        session_id = "default"

        if is_alice:
            req = data.get("request", {})
            session = data.get("session", {})

            session_id = session.get("session_id", "default")

            text = (
                req.get("command")
                or req.get("original_utterance")
                or ""
            ).strip()

        else:
            text = data.get("text", "").strip()

        print(
            f"SESSION: {session_id} | INPUT: {text}",
            flush=True
        )

        if not text:
            answer = "Привет! Чем могу помочь?"

        else:
            history = sessions.get(session_id, [])

            messages = [
                {
                    "role": "system",
                    "content": (
                        "Ты голосовой помощник Эл. "
                        "Ты работаешь через Яндекс Алису, но ответы формируешь сам. "
                        "Отвечай по-русски естественно, умно и по существу. "
                        "Обязательно учитывай предыдущие реплики диалога. "
                        "Если пользователь говорит 'он', 'она', 'у неё', "
                        "'у него', 'это', 'там' и подобные слова, "
                        "определи, к чему они относятся из предыдущего контекста. "
                        "Не задавай уточняющий вопрос, если смысл очевиден из истории. "
                        "Для голосового ответа обычно используй 1-3 коротких предложения. "
                        "Не используй markdown, таблицы и длинные списки."
                    )
                }
            ]

            messages.extend(history)

            messages.append({
                "role": "user",
                "content": text
            })

            openai_started = time.time()

            response = client.responses.create(
                model="gpt-5-mini",
                input=messages,
                reasoning={
                    "effort": "low"
                },
                max_output_tokens=140
            )

            answer = response.output_text.strip()

            openai_time = time.time() - openai_started

            print(
                f"OPENAI TIME: {openai_time:.2f}s",
                flush=True
            )

            if not answer:
                answer = "Не удалось сформировать ответ."

            # Добавляем текущий обмен в память
            history.append({
                "role": "user",
                "content": text
            })

            history.append({
                "role": "assistant",
                "content": answer
            })

            # Оставляем только последние сообщения,
            # чтобы не раздувать контекст и не замедлять ответы
            history = history[-MAX_HISTORY_MESSAGES:]

            sessions[session_id] = history

        # Для голосового интерфейса держим ответ коротким
        answer = answer[:900]

        total_time = time.time() - started

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

        data = request.get_json(silent=True) or {}

        is_alice = (
            "request" in data
            and "session" in data
        )

        if is_alice:
            fallback = (
                "Ответ занял слишком много времени. "
                "Попробуйте повторить вопрос."
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

    app.run(
        host="0.0.0.0",
        port=port
    )
