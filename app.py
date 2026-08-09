import os
import time
import re
import threading
import uuid

from flask import Flask, request, jsonify
from openai import OpenAI, APITimeoutError, APIError


app = Flask(__name__)

MODEL_NAME = "gpt-5.6-luna"


# =========================================================
# TIMING
# =========================================================

# Сколько максимум сам webhook ждёт OpenAI.
# После этого Алисе отдаём локальный ответ,
# а OpenAI продолжает работать в фоне.
ALICE_WAIT_SECONDS = 3.2

# Сам фоновый OpenAI-запрос может работать дольше.
BACKGROUND_OPENAI_TIMEOUT = 15.0

# Продолжение модели тоже запускаем в фоне.
BACKGROUND_CONTINUATION_TIMEOUT = 15.0


# =========================================================
# OPENAI CLIENT
# =========================================================

client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
    timeout=BACKGROUND_OPENAI_TIMEOUT,
    max_retries=0
)


# =========================================================
# STARTUP WARMUP
# =========================================================

_warmup_started = False
_warmup_lock = threading.Lock()


def warmup_openai():
    global _warmup_started

    with _warmup_lock:
        if _warmup_started:
            return

        _warmup_started = True

    time.sleep(3.0)

    started = time.time()

    try:
        print(
            "OPENAI WARMUP START",
            flush=True
        )

        response = client.responses.create(
            model=MODEL_NAME,
            input="Ответь только: OK",
            reasoning={
                "effort": "none"
            },
            max_output_tokens=16,
            timeout=10.0
        )

        print(
            f"OPENAI WARMUP OK | "
            f"TIME: {time.time() - started:.2f}s | "
            f"STATUS: {getattr(response, 'status', None)}",
            flush=True
        )

    except Exception as e:
        print(
            f"OPENAI WARMUP FAILED | "
            f"TIME: {time.time() - started:.2f}s | "
            f"{type(e).__name__}: {e}",
            flush=True
        )


threading.Thread(
    target=warmup_openai,
    name="openai-warmup",
    daemon=True
).start()


# =========================================================
# CONFIG
# =========================================================

sessions = {}

MAX_HISTORY_ITEMS = 10

CHUNK_TARGET = 700
CHUNK_HARD_MAX = 900

CONTINUE_PROMPT = "Продолжить?"


# =========================================================
# SYSTEM PROMPT
# =========================================================

SYSTEM_PROMPT = """
Ты голосовой помощник Эл.

Ты работаешь через голосовой интерфейс Алисы,
но содержательные ответы формируешь самостоятельно.

Отвечай по-русски, естественно, точно и по существу.

Всегда учитывай предыдущий контекст разговора.

Если пользователь говорит "он", "она", "у него", "у неё",
"ему", "ей", "это", "там", "тот", "эта" и подобные слова,
определи объект из предыдущих реплик.

Простой фактический вопрос — краткий прямой ответ.

Обычное объяснение — законченный ответ средней длины.

Анализ, прогноз, сравнение, маршрут или подробный рассказ —
развёрнутый законченный ответ.

Не пытайся обязательно вместить большой ответ в один короткий блок.
Backend самостоятельно разбивает длинные ответы и умеет продолжать их.

Пиши нормальными законченными предложениями.
Старайся делать отдельное предложение короче 300 символов.

Не используй Markdown, таблицы и сложное форматирование.
Не называй себя Алисой.
Избегай воды и повторов.
"""


# =========================================================
# LOCAL COMMANDS
# =========================================================

CONTINUE_WORDS = {
    "продолжай",
    "продолжить",
    "дальше",
    "еще",
    "ещё",
    "да",
    "давай дальше",
    "продолжай дальше",
    "рассказывай дальше",
    "есть что еще",
    "есть что ещё",
    "а еще",
    "а ещё",
    "что еще",
    "что ещё"
}


DONE_CHECK_WORDS = {
    "все",
    "всё",
    "это все",
    "это всё",
    "это вся информация",
    "это вся информация по текущему запросу",
    "это все по текущему запросу",
    "это всё по текущему запросу",
    "это полный ответ",
    "больше ничего"
}


THANKS_WORDS = {
    "спасибо",
    "ок спасибо",
    "хорошо спасибо",
    "понял спасибо",
    "понятно спасибо",
    "благодарю",
    "спасибо эл",
    "окей спасибо"
}


# =========================================================
# TEXT HELPERS
# =========================================================

def normalize_command(text):
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


# =========================================================
# INITIAL OUTPUT BUDGET
# =========================================================

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
        "посмотри маршрут",
        "что можешь сказать",
        "плюсы и минусы",
        "пошагово",
        "куда лучше",
        "как добраться"
    )

    if any(marker in t for marker in long_markers):
        return 220

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

    if any(marker in t for marker in medium_markers):
        return 180

    short_markers = (
        "сколько",
        "когда",
        "кто",
        "где",
        "какая столица",
        "сколько лет"
    )

    if any(marker in t for marker in short_markers):
        return 120

    return 160


# =========================================================
# SENTENCE CHUNKING
# =========================================================

def split_sentences(text):
    text = re.sub(
        r"\s+",
        " ",
        text or ""
    ).strip()

    if not text:
        return []

    return [
        part.strip()
        for part in re.split(
            r'(?<=[.!?…])\s+',
            text
        )
        if part.strip()
    ]


def split_oversized_sentence(sentence):
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
            chunks.append(current.strip())

        current = part

    if current:
        chunks.append(current.strip())

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

            final_chunks.append(
                remaining[:cut].strip()
            )

            remaining = remaining[cut:].strip()

        if remaining:
            final_chunks.append(
                remaining
            )

    return final_chunks


def split_into_chunks(text):
    sentences = split_sentences(text)

    if not sentences:
        return [text.strip()] if text.strip() else []

    normalized = []

    for sentence in sentences:
        normalized.extend(
            split_oversized_sentence(sentence)
        )

    chunks = []
    current = ""

    for sentence in normalized:
        candidate = (
            sentence
            if not current
            else current + " " + sentence
        )

        if len(candidate) <= CHUNK_TARGET:
            current = candidate

        else:
            if current:
                chunks.append(current.strip())

            current = sentence.strip()

    if current:
        chunks.append(current.strip())

    return chunks


# =========================================================
# OPENAI RESPONSE STATE
# =========================================================

def get_incomplete_reason(response):
    if getattr(
        response,
        "status",
        None
    ) != "incomplete":
        return None

    details = getattr(
        response,
        "incomplete_details",
        None
    )

    if not details:
        return None

    return getattr(
        details,
        "reason",
        None
    )


def response_needs_continuation(response):
    return (
        getattr(
            response,
            "status",
            None
        ) == "incomplete"
        and
        get_incomplete_reason(
            response
        ) == "max_output_tokens"
    )


# =========================================================
# SESSION
# =========================================================

def get_session(session_id):
    if session_id not in sessions:
        sessions[session_id] = {
            "history": [],
            "pending_chunks": [],
            "previous_response_id": None,
            "needs_model_continuation": False,

            # Фоновая генерация
            "generation_running": False,
            "generation_ready": False,
            "generation_job_id": None,
            "generation_error": None,

            # Защита состояния от нескольких потоков
            "lock": threading.Lock()
        }

    return sessions[session_id]


# =========================================================
# ALICE RESPONSE
# =========================================================

def alice_response(
    text,
    has_more=False,
    model_can_continue=False
):
    spoken_text = text.strip()

    should_offer_continue = (
        has_more
        or model_can_continue
    )

    if should_offer_continue:
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

    if should_offer_continue:
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


# =========================================================
# STORE FINISHED MODEL RESPONSE
# =========================================================

def store_model_result(
    state,
    job_id,
    user_text,
    history_before,
    response
):
    full_answer = (
        response.output_text
        or ""
    ).strip()

    response_status = getattr(
        response,
        "status",
        None
    )

    incomplete_reason = (
        get_incomplete_reason(
            response
        )
    )

    chunks = split_into_chunks(
        full_answer
    )

    if not chunks:
        chunks = [
            "Не удалось сформировать ответ."
        ]

    with state["lock"]:

        # Если пользователь уже задал новый вопрос,
        # старый фоновый результат игнорируем.
        if (
            state["generation_job_id"]
            != job_id
        ):
            print(
                f"BACKGROUND RESULT DISCARDED | "
                f"JOB: {job_id}",
                flush=True
            )
            return

        state["pending_chunks"] = chunks

        state["previous_response_id"] = (
            getattr(
                response,
                "id",
                None
            )
        )

        state[
            "needs_model_continuation"
        ] = response_needs_continuation(
            response
        )

        history = list(
            history_before
        )

        history.append({
            "role": "user",
            "content": user_text
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

        state["generation_running"] = False
        state["generation_ready"] = True
        state["generation_error"] = None

    print(
        f"BACKGROUND READY | "
        f"JOB: {job_id} | "
        f"LENGTH: {len(full_answer)} | "
        f"STATUS: {response_status} | "
        f"INCOMPLETE: {incomplete_reason} | "
        f"CHUNKS: {[len(x) for x in chunks]}",
        flush=True
    )


# =========================================================
# INITIAL BACKGROUND GENERATION
# =========================================================

def run_background_generation(
    state,
    job_id,
    user_text,
    conversation,
    history_before,
    output_budget,
    finished_event
):
    started = time.time()

    try:
        print(
            f"BACKGROUND OPENAI START | "
            f"JOB: {job_id} | "
            f"BUDGET: {output_budget}",
            flush=True
        )

        response = client.responses.create(
            model=MODEL_NAME,
            instructions=SYSTEM_PROMPT,
            input=conversation,
            reasoning={
                "effort": "none"
            },
            max_output_tokens=output_budget,
            timeout=BACKGROUND_OPENAI_TIMEOUT
        )

        print(
            f"BACKGROUND OPENAI TIME: "
            f"{time.time() - started:.2f}s | "
            f"JOB: {job_id}",
            flush=True
        )

        store_model_result(
            state=state,
            job_id=job_id,
            user_text=user_text,
            history_before=history_before,
            response=response
        )

    except Exception as e:
        with state["lock"]:
            if (
                state["generation_job_id"]
                == job_id
            ):
                state["generation_running"] = False
                state["generation_ready"] = False
                state["generation_error"] = (
                    f"{type(e).__name__}: {e}"
                )

        print(
            f"BACKGROUND OPENAI ERROR | "
            f"JOB: {job_id} | "
            f"TIME: {time.time() - started:.2f}s | "
            f"{type(e).__name__}: {e}",
            flush=True
        )

    finally:
        finished_event.set()


# =========================================================
# BACKGROUND MODEL CONTINUATION
# =========================================================

def run_background_continuation(
    state,
    job_id,
    previous_response_id,
    finished_event
):
    started = time.time()

    try:
        print(
            f"BACKGROUND CONTINUATION START | "
            f"JOB: {job_id}",
            flush=True
        )

        response = client.responses.create(
            model=MODEL_NAME,

            instructions=SYSTEM_PROMPT,

            previous_response_id=(
                previous_response_id
            ),

            input=(
                "Продолжи предыдущий ответ точно "
                "с того места, где он оборвался. "
                "Не повторяй уже сказанное."
            ),

            reasoning={
                "effort": "none"
            },

            max_output_tokens=420,

            timeout=(
                BACKGROUND_CONTINUATION_TIMEOUT
            )
        )

        full_answer = (
            response.output_text
            or ""
        ).strip()

        chunks = split_into_chunks(
            full_answer
        )

        if not chunks:
            chunks = [
                "Не удалось сформировать продолжение."
            ]

        with state["lock"]:

            if (
                state["generation_job_id"]
                != job_id
            ):
                print(
                    f"BACKGROUND CONTINUATION "
                    f"DISCARDED | JOB: {job_id}",
                    flush=True
                )
                return

            state["pending_chunks"] = chunks

            state[
                "previous_response_id"
            ] = getattr(
                response,
                "id",
                None
            )

            state[
                "needs_model_continuation"
            ] = response_needs_continuation(
                response
            )

            state["generation_running"] = False
            state["generation_ready"] = True
            state["generation_error"] = None

        print(
            f"BACKGROUND CONTINUATION READY | "
            f"JOB: {job_id} | "
            f"TIME: {time.time() - started:.2f}s | "
            f"CHUNKS: {[len(x) for x in chunks]} | "
            f"MODEL_CONTINUE: "
            f"{response_needs_continuation(response)}",
            flush=True
        )

    except Exception as e:

        with state["lock"]:
            if (
                state["generation_job_id"]
                == job_id
            ):
                state["generation_running"] = False
                state["generation_ready"] = False
                state["generation_error"] = (
                    f"{type(e).__name__}: {e}"
                )

        print(
            f"BACKGROUND CONTINUATION ERROR | "
            f"JOB: {job_id} | "
            f"TIME: {time.time() - started:.2f}s | "
            f"{type(e).__name__}: {e}",
            flush=True
        )

    finally:
        finished_event.set()


# =========================================================
# HEALTH
# =========================================================

@app.get("/")
def health():
    return jsonify({
        "status": "ok",
        "service": "alice-el-backend"
    })


# =========================================================
# MAIN WEBHOOK
# =========================================================

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
        # REQUEST PARSING
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
            f"SESSION: {session_id} | "
            f"INPUT: {text}",
            flush=True
        )


        state = get_session(
            session_id
        )

        normalized_text = (
            normalize_command(
                text
            )
        )

        wants_continue = (
            continue_by_button
            or normalized_text
            in CONTINUE_WORDS
        )


        # =================================================
        # LOCAL DONE CHECK
        # =================================================

        if normalized_text in DONE_CHECK_WORDS:

            with state["lock"]:
                generation_running = (
                    state[
                        "generation_running"
                    ]
                )

                has_more = bool(
                    state[
                        "pending_chunks"
                    ]
                )

                model_can_continue = (
                    state[
                        "needs_model_continuation"
                    ]
                )

            if generation_running:
                answer = (
                    "Ответ ещё готовится. "
                    "Скажите «продолжай» через несколько секунд."
                )

            elif (
                has_more
                or model_can_continue
            ):
                answer = (
                    "Нет, информация ещё осталась."
                )

            else:
                answer = (
                    "Да, это полный ответ "
                    "по текущему запросу."
                )

            if is_alice:
                return alice_response(
                    answer,
                    has_more=has_more,
                    model_can_continue=model_can_continue
                )

            return jsonify({
                "answer": answer
            })


        # =================================================
        # LOCAL THANKS
        # =================================================

        if normalized_text in THANKS_WORDS:

            if is_alice:
                return alice_response(
                    "Пожалуйста!"
                )

            return jsonify({
                "answer": "Пожалуйста!"
            })


        # =================================================
        # CONTINUE WHILE BACKGROUND IS STILL RUNNING
        # =================================================

        if wants_continue:

            with state["lock"]:
                generation_running = (
                    state[
                        "generation_running"
                    ]
                )

                generation_error = (
                    state[
                        "generation_error"
                    ]
                )


            if generation_running:

                answer = (
                    "Ответ ещё готовится. "
                    "Скажите «продолжай» "
                    "ещё через несколько секунд."
                )

                print(
                    "BACKGROUND STILL RUNNING",
                    flush=True
                )

                if is_alice:
                    return alice_response(
                        answer
                    )

                return jsonify({
                    "answer": answer
                })


            if generation_error:

                answer = (
                    "Не удалось подготовить ответ. "
                    "Повторите вопрос."
                )

                print(
                    "BACKGROUND RESULT ERROR: "
                    f"{generation_error}",
                    flush=True
                )

                if is_alice:
                    return alice_response(
                        answer
                    )

                return jsonify({
                    "answer": answer
                })


        # =================================================
        # READY PENDING CHUNK
        # =================================================

        if wants_continue:

            with state["lock"]:

                if state["pending_chunks"]:

                    next_chunk = (
                        state[
                            "pending_chunks"
                        ].pop(0)
                    )

                    has_more = bool(
                        state[
                            "pending_chunks"
                        ]
                    )

                    model_can_continue = (
                        not has_more
                        and state[
                            "needs_model_continuation"
                        ]
                    )

                else:
                    next_chunk = None
                    has_more = False
                    model_can_continue = False


            if next_chunk is not None:

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
                        has_more=has_more,
                        model_can_continue=model_can_continue
                    )

                return jsonify({
                    "answer": next_chunk,
                    "has_more": has_more,
                    "model_can_continue":
                        model_can_continue
                })


        # =================================================
        # MODEL CONTINUATION NEEDED
        # =================================================

        if wants_continue:

            with state["lock"]:
                needs_model_continuation = (
                    state[
                        "needs_model_continuation"
                    ]
                )

                previous_response_id = (
                    state[
                        "previous_response_id"
                    ]
                )


            if (
                needs_model_continuation
                and previous_response_id
            ):

                job_id = str(
                    uuid.uuid4()
                )

                finished_event = (
                    threading.Event()
                )

                with state["lock"]:
                    state[
                        "generation_job_id"
                    ] = job_id

                    state[
                        "generation_running"
                    ] = True

                    state[
                        "generation_ready"
                    ] = False

                    state[
                        "generation_error"
                    ] = None


                thread = threading.Thread(
                    target=run_background_continuation,
                    args=(
                        state,
                        job_id,
                        previous_response_id,
                        finished_event
                    ),
                    name=(
                        f"continuation-{job_id[:8]}"
                    ),
                    daemon=True
                )

                thread.start()


                # Даём модели шанс ответить быстро.
                finished_event.wait(
                    ALICE_WAIT_SECONDS
                )


                with state["lock"]:
                    ready = (
                        state[
                            "generation_ready"
                        ]
                    )


                if ready:

                    with state["lock"]:

                        if state["pending_chunks"]:

                            first_chunk = (
                                state[
                                    "pending_chunks"
                                ].pop(0)
                            )

                            has_more = bool(
                                state[
                                    "pending_chunks"
                                ]
                            )

                            model_can_continue = (
                                not has_more
                                and state[
                                    "needs_model_continuation"
                                ]
                            )

                        else:
                            first_chunk = None
                            has_more = False
                            model_can_continue = False


                    if first_chunk:

                        if is_alice:
                            return alice_response(
                                first_chunk,
                                has_more=has_more,
                                model_can_continue=model_can_continue
                            )

                        return jsonify({
                            "answer": first_chunk
                        })


                print(
                    "CONTINUATION DEFERRED TO BACKGROUND",
                    flush=True
                )

                answer = (
                    "Продолжение готовится. "
                    "Скажите «продолжай» "
                    "через несколько секунд."
                )

                if is_alice:
                    return alice_response(
                        answer
                    )

                return jsonify({
                    "answer": answer
                })


        # =================================================
        # NOTHING LEFT TO CONTINUE
        # =================================================

        if wants_continue:

            answer = (
                "Это был полный ответ. "
                "Можете задать следующий вопрос."
            )

            print(
                "NO MORE CONTENT",
                flush=True
            )

            if is_alice:
                return alice_response(
                    answer
                )

            return jsonify({
                "answer": answer
            })


        # =================================================
        # NEW QUESTION
        # =================================================

        if text:

            # Новый вопрос делает старую фоновую
            # задачу логически неактуальной.
            new_job_id = str(
                uuid.uuid4()
            )

            with state["lock"]:

                if state["pending_chunks"]:
                    print(
                        "OLD PENDING CHUNKS CLEARED",
                        flush=True
                    )

                state[
                    "pending_chunks"
                ] = []

                state[
                    "previous_response_id"
                ] = None

                state[
                    "needs_model_continuation"
                ] = False

                state[
                    "generation_job_id"
                ] = new_job_id

                state[
                    "generation_running"
                ] = False

                state[
                    "generation_ready"
                ] = False

                state[
                    "generation_error"
                ] = None


        # =================================================
        # EMPTY INPUT
        # =================================================

        if not text:

            if is_alice:
                return alice_response(
                    "Привет! Я Эл. Чем могу помочь?"
                )

            return jsonify({
                "answer":
                    "Привет! Я Эл. Чем могу помочь?"
            })


        # =================================================
        # QUICK LOCAL RESPONSE
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
        # PREPARE MODEL REQUEST
        # =================================================

        with state["lock"]:
            history_before = list(
                state["history"][
                    -MAX_HISTORY_ITEMS:
                ]
            )


        conversation = []

        for item in history_before:
            conversation.append({
                "role": item["role"],
                "content": item["content"]
            })

        conversation.append({
            "role": "user",
            "content": text
        })


        output_budget = (
            choose_output_budget(
                text
            )
        )


        job_id = str(
            uuid.uuid4()
        )

        finished_event = (
            threading.Event()
        )


        with state["lock"]:

            state[
                "generation_job_id"
            ] = job_id

            state[
                "generation_running"
            ] = True

            state[
                "generation_ready"
            ] = False

            state[
                "generation_error"
            ] = None


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
        # START BACKGROUND OPENAI
        # =================================================

        thread = threading.Thread(
            target=run_background_generation,
            args=(
                state,
                job_id,
                text,
                conversation,
                history_before,
                output_budget,
                finished_event
            ),
            name=(
                f"openai-{job_id[:8]}"
            ),
            daemon=True
        )

        thread.start()


        # =================================================
        # WAIT ONLY FOR ALICE-SAFE WINDOW
        # =================================================

        finished_event.wait(
            ALICE_WAIT_SECONDS
        )


        with state["lock"]:

            ready = (
                state[
                    "generation_ready"
                ]
            )

            error = (
                state[
                    "generation_error"
                ]
            )


        # =================================================
        # MODEL FINISHED WITHIN ~3.2s
        # =================================================

        if ready:

            with state["lock"]:

                if state["pending_chunks"]:

                    first_chunk = (
                        state[
                            "pending_chunks"
                        ].pop(0)
                    )

                    has_more = bool(
                        state[
                            "pending_chunks"
                        ]
                    )

                    model_can_continue = (
                        not has_more
                        and state[
                            "needs_model_continuation"
                        ]
                    )

                else:
                    first_chunk = None
                    has_more = False
                    model_can_continue = False


            if first_chunk:

                total_time = (
                    time.time()
                    - started
                )

                print(
                    f"FAST BACKGROUND RESULT | "
                    f"TOTAL TIME: {total_time:.2f}s",
                    flush=True
                )

                if is_alice:
                    return alice_response(
                        first_chunk,
                        has_more=has_more,
                        model_can_continue=model_can_continue
                    )

                return jsonify({
                    "answer": first_chunk,
                    "has_more": has_more,
                    "model_can_continue":
                        model_can_continue
                })


        # =================================================
        # BACKGROUND FAILED VERY QUICKLY
        # =================================================

        if error:

            answer = (
                "Сейчас не удалось получить ответ. "
                "Повторите вопрос."
            )

            if is_alice:
                return alice_response(
                    answer
                )

            return jsonify({
                "answer": answer
            })


        # =================================================
        # OPENAI IS STILL WORKING
        #
        # IMPORTANT:
        # request continues in background.
        # =================================================

        total_time = (
            time.time()
            - started
        )

        print(
            f"ALICE DEADLINE REACHED | "
            f"JOB CONTINUES IN BACKGROUND | "
            f"TOTAL TIME: {total_time:.2f}s",
            flush=True
        )


        answer = (
            "Готовлю ответ. "
            "Скажите «продолжай» "
            "через несколько секунд."
        )


        if is_alice:
            return alice_response(
                answer
            )


        return jsonify({
            "answer": answer,
            "background": True
        })


    # =====================================================
    # FATAL ERROR
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


# =========================================================
# LOCAL START
# =========================================================

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
