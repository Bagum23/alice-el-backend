import os
import time
import re

from flask import Flask, request, jsonify
from openai import OpenAI, APITimeoutError, APIError


app = Flask(__name__)

client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
    timeout=4.5,
    max_retries=0
)

# Память диалога по session_id Алисы
sessions = {}


SYSTEM_PROMPT = """
Ты голосовой помощник Эл.

Ты работаешь внутри голосового интерфейса Алисы,
но ответы пользователю формируешь ты.

Отвечай по-русски, естественно, точно и по существу.

Выбирай длину ответа по смыслу вопроса.

На простой фактический вопрос отвечай коротко и прямо,
обычно одним-двумя предложениями.

На вопросы, требующие объяснения, давай законченный ответ
с достаточным количеством деталей.

Если пользователь просит анализ, прогноз, сравнение,
развёрнутое объяснение, историю или подробный рассказ,
отвечай подробнее, обычно до 150-250 слов.

Не сокращай ответ настолько, чтобы пользователю приходилось
просить "продолжай" для получения основной информации.

Если пользователь говорит:
"короче" — сократи следующий ответ;
"подробнее" — дай больше деталей;
"объясни проще" — объясни простыми словами;
"продолжай" — продолжи предыдущую мысль без повторения начала.

Всегда учитывай предыдущие реплики диалога.

Если пользователь говорит "он", "она", "у него", "у неё",
"это", "там", "тот", "эта", "ему", "ей" и подобные слова,
определи, к чему они относятся из предыдущего контекста.

Не задавай уточняющий вопрос, если смысл очевиден
из предыдущих реплик.

Не используй Markdown, таблицы и служебные пояснения.

Не говори, что ты Алиса.

Избегай лишней воды и повторов.
"""


def alice_response(text, end_session=False):
    return jsonify({
        "response": {
            "text": text,
            "tts": text,
            "end_session": end_session
        },
        "version": "1.0"
    })


def quick_answer(text):
    """
    Мгновенные локальные ответы.
    OpenAI для них не вызывается.
    """

    t = text.lower().strip()

    greetings = {
        "привет",
        "здравствуй",
        "здравствуйте",
        "добрый день",
        "добрый вечер",
        "доброе утро",
        "привет эл",
        "эл привет"
    }

    if t in greetings:
        return "Привет! Я Эл. Чем могу помочь?"

    multiplication = re.fullmatch(
        r"\s*(-?\d+(?:[.,]\d+)?)\s*"
        r"(?:умножить\s+на|×|\*)\s*"
        r"(-?\d+(?:[.,]\d+)?)\s*",
        t
    )

    if multiplication:
        try:
            a = float(multiplication.group(1).replace(",", "."))
            b = float(multiplication.group(2).replace(",", "."))
            result = a * b

            if result.is_integer():
                result = int(result)

            return (
                f"{multiplication.group(1)} умножить на "
                f"{multiplication.group(2)} равно {result}."
            )

        except Exception:
            pass

    return None


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
            isinstance(data, dict)
            and "request" in data
            and "session" in data
        )

        if is_alice:
            req = data.get("request", {})
            session = data.get("session", {})

            text = (
                req.get("original_utterance")
                or req.get("command")
                or ""
            ).strip()

            session_id = session.get(
                "session_id",
                "default"
            )

        else:
            text = str(
                data.get("text", "")
            ).strip()

            session_id = str(
                data.get("session_id", "default")
            )

        print(
            f"SESSION: {session_id} | INPUT: {text}",
            flush=True
        )

        # Пустой запрос
        if not text:
            answer = "Привет! Я Эл. Чем могу помочь?"

            if is_alice:
                return alice_response(answer)

            return jsonify({
                "answer": answer
            })

        # Быстрый локальный ответ
        answer = quick_answer(text)

        if answer:
            print(
                "LOCAL ANSWER",
                flush=True
            )

            if is_alice:
                return alice_response(answer)

            return jsonify({
                "answer": answer
            })

        # История текущей сессии
        history = sessions.get(
            session_id,
            []
        )

        # Держим несколько последних реплик.
        # Этого достаточно для нормального контекста
        # и не слишком раздувает запрос.
        history = history[-8:]

        conversation = []

        for item in history:
            conversation.append({
                "role": item["role"],
                "content": item["content"]
            })

        conversation.append({
            "role": "user",
            "content": text
        })

        openai_started = time.time()

        try:
            response = client.responses.create(
                model="gpt-5.5",
                instructions=SYSTEM_PROMPT,
                input=conversation,
                reasoning={
                    "effort": "low"
                },
                max_output_tokens=400,
                timeout=4.5
            )

            answer = (
                response.output_text
                or ""
            ).strip()

            print(
                f"OPENAI TIME: "
                f"{time.time() - openai_started:.2f}s",
                flush=True
            )

        except APITimeoutError:
            print(
                f"OPENAI TIMEOUT after "
                f"{time.time() - openai_started:.2f}s",
                flush=True
            )

            answer = (
                "Ответ занял слишком много времени. "
                "Повторите вопрос."
            )

        except APIError as e:
            print(
                f"OPENAI API ERROR: "
                f"{type(e).__name__}: {e}",
                flush=True
            )

            answer = (
                "Сейчас не удалось получить ответ. "
                "Повторите вопрос."
            )

        except Exception as e:
            print(
                f"OPENAI ERROR: "
                f"{type(e).__name__}: {e}",
                flush=True
            )

            answer = (
                "Сейчас не удалось получить ответ. "
                "Повторите вопрос."
            )

        if not answer:
            answer = (
                "Не удалось сформировать ответ. "
                "Повторите вопрос."
            )

        # Сохраняем только сам разговор
        history.append({
            "role": "user",
            "content": text
        })

        history.append({
            "role": "assistant",
            "content": answer
        })

        sessions[session_id] = history[-10:]

        total_time = time.time() - started

        print(
            f"TOTAL TIME: {total_time:.2f}s",
            flush=True
        )

        if is_alice:
            return alice_response(answer)

        return jsonify({
            "answer": answer
        })

    except Exception as e:
        print(
            f"FATAL ERROR: "
            f"{type(e).__name__}: {e}",
            flush=True
        )

        try:
            data = request.get_json(
                silent=True
            ) or {}

            if (
                "request" in data
                and "session" in data
            ):
                return alice_response(
                    "Произошла ошибка. Повторите вопрос."
                )

        except Exception:
            pass

        return jsonify({
            "answer": "Произошла ошибка. Повторите вопрос."
        })


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
