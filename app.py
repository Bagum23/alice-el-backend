import os
import time
import re

from flask import Flask, request, jsonify
from openai import OpenAI, APITimeoutError, APIError


app = Flask(__name__)

client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
    timeout=5.5,
    max_retries=0
)

sessions = {}


SYSTEM_PROMPT = """
Ты голосовой помощник Эл.

Ты работаешь внутри голосового интерфейса Алисы,
но ответы формируешь самостоятельно через OpenAI.

Отвечай по-русски естественно, точно и по существу.

Всегда учитывай предыдущий контекст разговора.

Если пользователь говорит "он", "она", "у него", "у неё",
"ему", "ей", "там", "это", "тот", "эта" и подобные слова,
определи, к чему они относятся из предыдущих реплик.

Не задавай уточняющий вопрос, если смысл очевиден из контекста.

Выбирай длину ответа по смыслу:
- простой факт — короткий прямой ответ;
- обычное объяснение — законченный ответ средней длины;
- анализ, прогноз, сравнение, подробный рассказ — более развёрнутый ответ.

Не сокращай основной ответ настолько, чтобы пользователю
приходилось говорить "продолжай", чтобы получить суть.

Если пользователь говорит "продолжай" — продолжай предыдущую
мысль без повторения начала.
Если говорит "подробнее" — добавь существенные детали.
Если говорит "короче" — дай краткое резюме.
Если говорит "объясни проще" — переформулируй простыми словами.

Не используй Markdown, таблицы и служебные пояснения.
Не называй себя Алисой.
Избегай воды и повторов.
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


def choose_output_budget(text):
    """
    Определяем необходимую длину ответа.
    Возвращаем max_output_tokens.
    """

    t = text.lower().strip()

    # Явно развёрнутые запросы
    long_markers = (
        "проанализируй",
        "анализ",
        "сравни",
        "сравнение",
        "прогноз",
        "спрогнозируй",
        "расскажи подробно",
        "расскажи подробнее",
        "объясни подробно",
        "подробно",
        "история",
        "перечисли основные",
        "что можешь сказать",
        "какая осень",
        "какой прогноз",
        "за последние",
        "плюсы и минусы"
    )

    if any(marker in t for marker in long_markers):
        return 420

    # Команды продолжения/расширения
    medium_long_markers = (
        "продолжай",
        "подробнее",
        "расскажи",
        "объясни",
        "почему",
        "как лучше",
        "что делать",
        "чем знаменит",
        "в какое время",
        "какие бывают"
    )

    if any(marker in t for marker in medium_long_markers):
        return 280

    # Короткий фактический вопрос
    short_markers = (
        "сколько",
        "когда",
        "кто",
        "где",
        "какая столица",
        "какой год",
        "сколько лет",
        "сколько ног",
        "сколько глаз"
    )

    if any(marker in t for marker in short_markers):
        return 140

    # Обычный вопрос
    return 220


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

            session_id = session.get("session_id", "default")

        else:
            text = str(data.get("text", "")).strip()
            session_id = str(data.get("session_id", "default"))

        print(
            f"SESSION: {session_id} | INPUT: {text}",
            flush=True
        )

        if not text:
            answer = "Привет! Я Эл. Чем могу помочь?"

            if is_alice:
                return alice_response(answer)

            return jsonify({"answer": answer})

        answer = quick_answer(text)

        if answer:
            print("LOCAL ANSWER", flush=True)

            if is_alice:
                return alice_response(answer)

            return jsonify({"answer": answer})

        history = sessions.get(session_id, [])
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

        output_budget = choose_output_budget(text)

        print(
            f"OUTPUT BUDGET: {output_budget}",
            flush=True
        )

        openai_started = time.time()

        try:
            response = client.responses.create(
                model="gpt-5.6-terra",
                instructions=SYSTEM_PROMPT,
                input=conversation,
                reasoning={
                    "effort": "low"
                },
                max_output_tokens=output_budget,
                timeout=5.5
            )

            answer = (response.output_text or "").strip()

            print(
                f"OPENAI TIME: "
                f"{time.time() - openai_started:.2f}s",
                flush=True
            )

        except APITimeoutError:
            elapsed = time.time() - openai_started

            print(
                f"OPENAI TIMEOUT after {elapsed:.2f}s",
                flush=True
            )

            answer = (
                "Ответ формируется дольше обычного. "
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
            data = request.get_json(silent=True) or {}

            if "request" in data and "session" in data:
                return alice_response(
                    "Произошла ошибка. Повторите вопрос."
                )

        except Exception:
            pass

        return jsonify({
            "answer": "Произошла ошибка. Повторите вопрос."
        })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))

    app.run(
        host="0.0.0.0",
        port=port
    )
