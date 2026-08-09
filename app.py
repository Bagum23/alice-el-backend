import os
import time
import re

from flask import Flask, request, jsonify
from openai import OpenAI, APITimeoutError, APIError


app = Flask(__name__)

MODEL_NAME = "gpt-5.6-luna"

client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
    timeout=4.2,
    max_retries=0
)

sessions = {}

MAX_HISTORY_ITEMS = 10

CHUNK_TARGET = 760
CHUNK_HARD_MAX = 900


SYSTEM_PROMPT = """
Ты голосовой помощник Эл.

Ты работаешь через голосовой интерфейс Алисы,
но ответы формируешь самостоятельно.

Отвечай по-русски, естественно, точно и по существу.

Всегда учитывай предыдущий контекст разговора.

Если пользователь говорит "он", "она", "у него", "у неё",
"ему", "ей", "это", "там", "тот", "эта" и подобные слова,
определи объект из предыдущих реплик.

Выбирай длину ответа по смыслу вопроса.

Простой факт — краткий прямой ответ.
Обычное объяснение — законченный ответ средней длины.
Анализ, прогноз, сравнение, маршрут или подробный рассказ —
развёрнутый законченный ответ.

Не сокращай содержательную часть только из-за голосового интерфейса:
backend самостоятельно разобьёт длинный текст на несколько голосовых частей.

Пиши нормальными законченными предложениями.
Не обрывай предложение посередине.

Не используй Markdown, таблицы и сложное форматирование.
Не называй себя Алисой.
Избегай воды и повторов.
"""


CONTINUE_WORDS = {
    "продолжай",
    "дальше",
    "еще",
    "ещё",
    "да",
    "продолжить",
    "рассказывай дальше"
}


def alice_response(text, has_more=False):
    response = {
        "text": text,
        "tts": text,
        "end_session": False
    }

    if has_more:
        response["buttons"] = [
            {
                "title": "Продолжить",
                "payload": {
                    "action": "continue"
                },
                "hide": True
            }
        ]

    return jsonify({
        "response": response,
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
    t = text.lower().strip()

    long_markers = (
        "проанализируй",
        "анализ",
        "сравни",
        "прогноз",
        "спрогнозируй",
        "расскажи подробно",
        "объясни подробно",
        "подробно",
        "за последние",
        "маршрут",
        "опиши маршрут",
        "что можешь сказать",
        "плюсы и минусы"
    )

    if any(x in t for x in long_markers):
        return 420

    medium_markers = (
        "расскажи",
        "объясни",
        "почему",
        "как лучше",
        "что делать",
        "чем знаменит",
        "какие места",
        "рекомендуешь"
    )

    if any(x in t for x in medium_markers):
        return 280

    short_markers = (
        "сколько",
        "когда",
        "кто",
        "где",
        "какая столица",
        "сколько лет"
    )

    if any(x in t for x in short_markers):
        return 160

    return 220


def split_sentences(text):
    text = re.sub(r"\s+", " ", text).strip()

    if not text:
        return []

    parts = re.split(
        r'(?<=[.!?…])\s+',
        text
    )

    return [
        part.strip()
        for part in parts
        if part.strip()
    ]


def split_into_chunks(text):
    sentences = split_sentences(text)

    if not sentences:
        return [text.strip()] if text.strip() else []

    chunks = []
    current = ""

    for sentence in sentences:
        candidate = (
            sentence
            if not current
            else current + " " + sentence
        )

        if len(candidate) <= CHUNK_TARGET:
            current = candidate
            continue

        if current:
            chunks.append(current)
            current = ""

        if len(sentence) > CHUNK_HARD_MAX:
            subparts = re.split(
                r'(?<=[;:,—])\s+',
                sentence
            )

            subcurrent = ""

            for part in subparts:
                candidate2 = (
                    part
                    if not subcurrent
                    else subcurrent + " " + part
                )

                if len(candidate2) <= CHUNK_HARD_MAX:
                    subcurrent = candidate2
                else:
                    if subcurrent:
                        chunks.append(subcurrent)

                    while len(part) > CHUNK_HARD_MAX:
                        cut = part.rfind(
                            " ",
                            0,
                            CHUNK_HARD_MAX
                        )

                        if cut < 200:
                            cut = CHUNK_HARD_MAX

                        chunks.append(
                            part[:cut].strip()
                        )

                        part = part[cut:].strip()

                    subcurrent = part

            if subcurrent:
                current = subcurrent

        else:
            current = sentence

    if current:
        chunks.append(current)

    return chunks


def get_session(session_id):
    if session_id not in sessions:
        sessions[session_id] = {
            "history": [],
            "pending_chunks": []
        }

    return sessions[session_id]


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

            session_id = session.get(
                "session_id",
                "default"
            )

            payload = req.get("payload") or {}

            text = (
                req.get("command")
                or req.get("original_utterance")
                or ""
            ).strip()

            continue_by_button = (
                payload.get("action") == "continue"
            )

        else:
            session_id = str(
                data.get("session_id", "manual")
            )

            text = str(
                data.get("text", "")
            ).strip()

            continue_by_button = False

        print(
            f"SESSION: {session_id} | INPUT: {text}",
            flush=True
        )

        state = get_session(session_id)

        wants_continue = (
            continue_by_button
            or text.lower().strip() in CONTINUE_WORDS
        )

        if wants_continue and state["pending_chunks"]:
            next_chunk = state["pending_chunks"].pop(0)

            has_more = bool(
                state["pending_chunks"]
            )

            print(
                f"PENDING CHUNK SENT | "
                f"REMAINING: {len(state['pending_chunks'])}",
                flush=True
            )

            if is_alice:
                return alice_response(
                    next_chunk,
                    has_more=has_more
                )

            return jsonify({
                "answer": next_chunk,
                "has_more": has_more
            })

        if text and not wants_continue:
            state["pending_chunks"] = []

        if not text:
            answer = "Привет! Я Эл. Чем могу помочь?"

            if is_alice:
                return alice_response(answer)

            return jsonify({
                "answer": answer
            })

        quick = quick_answer(text)

        if quick:
            print(
                "LOCAL ANSWER",
                flush=True
            )

            if is_alice:
                return alice_response(quick)

            return jsonify({
                "answer": quick
            })

        history = state["history"][-MAX_HISTORY_ITEMS:]

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
            f"MODEL: {MODEL_NAME}",
            flush=True
        )

        print(
            f"OUTPUT BUDGET: {output_budget}",
            flush=True
        )

        openai_started = time.time()

        try:
            response = client.responses.create(
                model=MODEL_NAME,
                instructions=SYSTEM_PROMPT,
                input=conversation,
                reasoning={
                    "effort": "none"
                },
                max_output_tokens=output_budget,
                timeout=4.2
            )

            full_answer = (
                response.output_text
                or ""
            ).strip()

            print(
                f"OPENAI TIME: "
                f"{time.time() - openai_started:.2f}s",
                flush=True
            )

            print(
                f"FULL ANSWER LENGTH: {len(full_answer)}",
                flush=True
            )

        except APITimeoutError:
            elapsed = (
                time.time()
                - openai_started
            )

            print(
                f"OPENAI TIMEOUT after {elapsed:.2f}s",
                flush=True
            )

            full_answer = (
                "Ответ формируется дольше обычного. "
                "Повторите вопрос."
            )

        except APIError as e:
            print(
                f"OPENAI API ERROR: "
                f"{type(e).__name__}: {e}",
                flush=True
            )

            full_answer = (
                "Сейчас не удалось получить ответ. "
                "Повторите вопрос."
            )

        except Exception as e:
            print(
                f"OPENAI ERROR: "
                f"{type(e).__name__}: {e}",
                flush=True
            )

            full_answer = (
                "Сейчас не удалось получить ответ. "
                "Повторите вопрос."
            )

        if not full_answer:
            full_answer = (
                "Не удалось сформировать ответ. "
                "Повторите вопрос."
            )

        history.append({
            "role": "user",
            "content": text
        })

        history.append({
            "role": "assistant",
            "content": full_answer
        })

        state["history"] = (
            history[-MAX_HISTORY_ITEMS:]
        )

        chunks = split_into_chunks(
            full_answer
        )

        if not chunks:
            chunks = [
                "Не удалось сформировать ответ."
            ]

        first_chunk = chunks[0]

        state["pending_chunks"] = (
            chunks[1:]
        )

        has_more = bool(
            state["pending_chunks"]
        )

        print(
            f"CHUNKS: {len(chunks)} | "
            f"FIRST LENGTH: {len(first_chunk)} | "
            f"REMAINING: {len(state['pending_chunks'])}",
            flush=True
        )

        total_time = (
            time.time()
            - started
        )

        print(
            f"TOTAL TIME: {total_time:.2f}s",
            flush=True
        )

        if is_alice:
            return alice_response(
                first_chunk,
                has_more=has_more
            )

        return jsonify({
            "answer": first_chunk,
            "has_more": has_more
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
