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

# ---------------------------------------------------------
# ПАМЯТЬ
# ---------------------------------------------------------

sessions = {}

MAX_HISTORY_ITEMS = 10

# Основной кусок держим существенно ниже лимита Алисы 1024.
CHUNK_TARGET = 700

# Даже одно длинное предложение стараемся не пропускать
# дальше этого размера.
CHUNK_HARD_MAX = 900

CONTINUE_PROMPT = "Продолжить?"


# ---------------------------------------------------------
# SYSTEM PROMPT
# ---------------------------------------------------------

SYSTEM_PROMPT = """
Ты голосовой помощник Эл.

Ты работаешь через голосовой интерфейс Алисы,
но содержательные ответы формируешь самостоятельно.

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
backend самостоятельно разобьёт длинный ответ на несколько частей.

Пиши нормальными законченными предложениями.

ВАЖНО:
старайся делать каждое отдельное предложение короче 300 символов.
Не создавай огромных предложений с множеством придаточных.

Не обрывай предложение посередине.

Не используй Markdown, таблицы и сложное форматирование.
Не называй себя Алисой.
Избегай воды и повторов.
"""


# ---------------------------------------------------------
# КОМАНДЫ ПРОДОЛЖЕНИЯ
# ---------------------------------------------------------

CONTINUE_WORDS = {
    "продолжай",
    "дальше",
    "еще",
    "ещё",
    "да",
    "продолжить",
    "рассказывай дальше",
    "давай дальше",
    "продолжай дальше"
}


def normalize_command(text):
    """
    Нормализуем голосовую команду:
    'Продолжай!' -> 'продолжай'
    '  Ещё? ' -> 'ещё'
    """

    text = (text or "").lower().strip()

    text = re.sub(
        r"[.!?,;:…]+$",
        "",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ---------------------------------------------------------
# ОТВЕТ АЛИСЕ
# ---------------------------------------------------------

def alice_response(text, has_more=False):
    """
    Если имеются следующие части, обязательно:
    1. добавляем голосом 'Продолжить?'
    2. показываем кнопку Продолжить.
    """

    spoken_text = text.strip()

    if has_more:
        spoken_text = (
            spoken_text.rstrip()
            + " "
            + CONTINUE_PROMPT
        )

    response = {
        "text": spoken_text,
        "tts": spoken_text,
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


# ---------------------------------------------------------
# БЫСТРЫЕ ЛОКАЛЬНЫЕ ОТВЕТЫ
# ---------------------------------------------------------

def quick_answer(text):

    t = normalize_command(text)

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
            a = float(
                multiplication.group(1).replace(",", ".")
            )

            b = float(
                multiplication.group(2).replace(",", ".")
            )

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


# ---------------------------------------------------------
# ДИНАМИЧЕСКАЯ ДЛИНА ОТВЕТА
# ---------------------------------------------------------

def choose_output_budget(text):

    t = normalize_command(text)

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
        "плюсы и минусы",
        "пошагово"
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
        "рекомендуешь",
        "куда лучше",
        "как добраться"
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


# ---------------------------------------------------------
# РАЗБИЕНИЕ НА ПРЕДЛОЖЕНИЯ
# ---------------------------------------------------------

def split_sentences(text):
    """
    Нормализуем пробелы и пытаемся делить строго
    после . ! ? …
    """

    text = re.sub(
        r"\s+",
        " ",
        text or ""
    ).strip()

    if not text:
        return []

    parts = re.split(
        r'(?<=[.!?…])\s+',
        text
    )

    result = []

    for part in parts:

        part = part.strip()

        if not part:
            continue

        result.append(part)

    return result


def ensure_sentence_end(text):
    """
    Крайний fallback.
    Если технически пришлось разрезать слишком длинную
    конструкцию, делаем получившийся фрагмент
    законченным для речи.
    """

    text = text.strip()

    if not text:
        return text

    if text[-1] not in ".!?…":
        text += "."

    return text


# ---------------------------------------------------------
# ОБРАБОТКА АНОМАЛЬНО ДЛИННОГО ПРЕДЛОЖЕНИЯ
# ---------------------------------------------------------

def split_oversized_sentence(sentence):
    """
    В норме сюда почти не попадём, поскольку SYSTEM_PROMPT
    просит предложения короче 300 символов.

    Если модель всё-таки сделала предложение > 900 символов,
    делим его сначала по смысловым паузам:
    ; : — ,

    Каждый получившийся речевой блок завершаем точкой.
    """

    if len(sentence) <= CHUNK_HARD_MAX:
        return [sentence.strip()]

    parts = re.split(
        r'(?<=[;:,—])\s+',
        sentence
    )

    chunks = []
    current = ""

    for part in parts:

        part = part.strip()

        if not part:
            continue

        candidate = (
            part
            if not current
            else current + " " + part
        )

        if len(candidate) <= CHUNK_TARGET:
            current = candidate
            continue

        if current:
            chunks.append(
                ensure_sentence_end(current)
            )

        current = part

    if current:
        chunks.append(
            ensure_sentence_end(current)
        )

    # Совсем крайний случай:
    # один кусок после разделения всё ещё огромный.
    final_chunks = []

    for chunk in chunks:

        if len(chunk) <= CHUNK_HARD_MAX:
            final_chunks.append(chunk)
            continue

        remaining = chunk

        while len(remaining) > CHUNK_HARD_MAX:

            cut = remaining.rfind(
                " ",
                0,
                CHUNK_TARGET
            )

            if cut < 200:
                cut = CHUNK_TARGET

            piece = remaining[:cut].strip()

            final_chunks.append(
                ensure_sentence_end(piece)
            )

            remaining = remaining[cut:].strip()

        if remaining:
            final_chunks.append(
                ensure_sentence_end(remaining)
            )

    return final_chunks


# ---------------------------------------------------------
# СОБИРАЕМ CHUNKS
# ---------------------------------------------------------

def split_into_chunks(text):
    """
    Главный принцип:

    - никогда намеренно не разрезаем нормальное предложение;
    - собираем несколько полных предложений до ~700 символов;
    - если следующее предложение не помещается —
      оно идёт в следующий chunk;
    - только патологически длинное предложение
      делится отдельным fallback-механизмом.
    """

    sentences = split_sentences(text)

    if not sentences:
        return [
            ensure_sentence_end(text)
        ] if text.strip() else []

    normalized_sentences = []

    for sentence in sentences:

        if len(sentence) <= CHUNK_HARD_MAX:
            normalized_sentences.append(
                sentence.strip()
            )
        else:
            normalized_sentences.extend(
                split_oversized_sentence(sentence)
            )

    chunks = []
    current = ""

    for sentence in normalized_sentences:

        sentence = sentence.strip()

        if not sentence:
            continue

        candidate = (
            sentence
            if not current
            else current + " " + sentence
        )

        if len(candidate) <= CHUNK_TARGET:

            current = candidate

        else:

            if current:
                chunks.append(
                    ensure_sentence_end(current)
                )

            current = sentence

    if current:
        chunks.append(
            ensure_sentence_end(current)
        )

    return chunks


# ---------------------------------------------------------
# SESSION
# ---------------------------------------------------------

def get_session(session_id):

    if session_id not in sessions:

        sessions[session_id] = {
            "history": [],
            "pending_chunks": []
        }

    return sessions[session_id]


# ---------------------------------------------------------
# HEALTH
# ---------------------------------------------------------

@app.get("/")
def health():

    return jsonify({
        "status": "ok",
        "service": "alice-el-backend"
    })


# ---------------------------------------------------------
# ОСНОВНОЙ WEBHOOK
# ---------------------------------------------------------

@app.post("/ask")
def ask():

    started = time.time()

    try:

        data = request.get_json(
            silent=True
        ) or {}

        is_alice = (
            isinstance(data, dict)
            and "request" in data
            and "session" in data
        )

        # -------------------------------------------------
        # Разбираем запрос Алисы
        # -------------------------------------------------

        if is_alice:

            req = data.get(
                "request",
                {}
            )

            session = data.get(
                "session",
                {}
            )

            session_id = session.get(
                "session_id",
                "default"
            )

            payload = req.get(
                "payload"
            ) or {}

            text = (
                req.get("command")
                or req.get("original_utterance")
                or ""
            ).strip()

            continue_by_button = (
                payload.get("action")
                == "continue"
            )

        # -------------------------------------------------
        # Обычный API-запрос для тестов
        # -------------------------------------------------

        else:

            session_id = str(
                data.get(
                    "session_id",
                    "manual"
                )
            )

            text = str(
                data.get(
                    "text",
                    ""
                )
            ).strip()

            continue_by_button = False


        print(
            f"SESSION: {session_id} | INPUT: {text}",
            flush=True
        )

        state = get_session(
            session_id
        )

        normalized_text = normalize_command(
            text
        )

        wants_continue = (
            continue_by_button
            or normalized_text in CONTINUE_WORDS
        )


        # =================================================
        # ПРОДОЛЖЕНИЕ ИЗ ГОТОВОЙ ОЧЕРЕДИ
        # =================================================

        if (
            wants_continue
            and state["pending_chunks"]
        ):

            next_chunk = (
                state["pending_chunks"].pop(0)
            )

            has_more = bool(
                state["pending_chunks"]
            )

            print(
                "PENDING CHUNK SENT | "
                f"LENGTH: {len(next_chunk)} | "
                f"REMAINING: "
                f"{len(state['pending_chunks'])}",
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


        # =================================================
        # Новый содержательный вопрос сбрасывает старый хвост
        # =================================================

        if (
            text
            and not wants_continue
        ):

            if state["pending_chunks"]:

                print(
                    "OLD PENDING CHUNKS CLEARED",
                    flush=True
                )

            state["pending_chunks"] = []


        # =================================================
        # ПУСТОЙ ЗАПРОС
        # =================================================

        if not text:

            answer = (
                "Привет! Я Эл. Чем могу помочь?"
            )

            if is_alice:

                return alice_response(
                    answer
                )

            return jsonify({
                "answer": answer
            })


        # =================================================
        # ЛОКАЛЬНЫЙ ОТВЕТ
        # =================================================

        quick = quick_answer(
            text
        )

        if quick:

            print(
                "LOCAL ANSWER",
                flush=True
            )

            if is_alice:

                return alice_response(
                    quick
                )

            return jsonify({
                "answer": quick
            })


        # =================================================
        # ИСТОРИЯ
        # =================================================

        history = (
            state["history"][
                -MAX_HISTORY_ITEMS:
            ]
        )

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


        # =================================================
        # ПАРАМЕТРЫ OPENAI
        # =================================================

        output_budget = (
            choose_output_budget(text)
        )

        print(
            f"MODEL: {MODEL_NAME}",
            flush=True
        )

        print(
            f"OUTPUT BUDGET: "
            f"{output_budget}",
            flush=True
        )


        # =================================================
        # OPENAI
        # =================================================

        openai_started = time.time()

        try:

            response = (
                client.responses.create(
                    model=MODEL_NAME,
                    instructions=SYSTEM_PROMPT,
                    input=conversation,
                    reasoning={
                        "effort": "none"
                    },
                    max_output_tokens=(
                        output_budget
                    ),
                    timeout=4.2
                )
            )

            full_answer = (
                response.output_text
                or ""
            ).strip()

            openai_time = (
                time.time()
                - openai_started
            )

            print(
                f"OPENAI TIME: "
                f"{openai_time:.2f}s",
                flush=True
            )

            print(
                "FULL ANSWER LENGTH: "
                f"{len(full_answer)}",
                flush=True
            )


        except APITimeoutError:

            elapsed = (
                time.time()
                - openai_started
            )

            print(
                "OPENAI TIMEOUT after "
                f"{elapsed:.2f}s",
                flush=True
            )

            full_answer = (
                "Ответ формируется дольше обычного. "
                "Повторите вопрос."
            )


        except APIError as e:

            print(
                "OPENAI API ERROR: "
                f"{type(e).__name__}: {e}",
                flush=True
            )

            full_answer = (
                "Сейчас не удалось получить ответ. "
                "Повторите вопрос."
            )


        except Exception as e:

            print(
                "OPENAI ERROR: "
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


        # =================================================
        # СОХРАНЯЕМ ПОЛНЫЙ ОТВЕТ В КОНТЕКСТ
        # =================================================

        history.append({
            "role": "user",
            "content": text
        })

        history.append({
            "role": "assistant",
            "content": full_answer
        })

        state["history"] = (
            history[
                -MAX_HISTORY_ITEMS:
            ]
        )


        # =================================================
        # РЕЖЕМ ПОЛНЫЙ ОТВЕТ НА ЗАКОНЧЕННЫЕ ФРАГМЕНТЫ
        # =================================================

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


        chunk_lengths = [
            len(chunk)
            for chunk in chunks
        ]

        print(
            f"CHUNKS: {len(chunks)} | "
            f"LENGTHS: {chunk_lengths} | "
            f"REMAINING: "
            f"{len(state['pending_chunks'])}",
            flush=True
        )


        # Дополнительная проверка для логов
        for index, chunk in enumerate(
            chunks,
            start=1
        ):

            ending = (
                chunk[-1]
                if chunk
                else ""
            )

            print(
                f"CHUNK {index} END: "
                f"{repr(ending)}",
                flush=True
            )


        total_time = (
            time.time()
            - started
        )

        print(
            f"TOTAL TIME: "
            f"{total_time:.2f}s",
            flush=True
        )


        # =================================================
        # ОТВЕТ
        # =================================================

        if is_alice:

            return alice_response(
                first_chunk,
                has_more=has_more
            )

        return jsonify({
            "answer": first_chunk,
            "has_more": has_more,
            "remaining_chunks": len(
                state["pending_chunks"]
            )
        })


    # =====================================================
    # FATAL
    # =====================================================

    except Exception as e:

        print(
            "FATAL ERROR: "
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
                    "Произошла ошибка. "
                    "Повторите вопрос."
                )

        except Exception:
            pass

        return jsonify({
            "answer":
                "Произошла ошибка. "
                "Повторите вопрос."
        })


# ---------------------------------------------------------
# START
# ---------------------------------------------------------

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
