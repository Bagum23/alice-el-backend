import os
import time
import re
import threading

from flask import Flask, request, jsonify
from openai import OpenAI, APITimeoutError, APIError


app = Flask(__name__)

MODEL_NAME = "gpt-5.6-luna"


# =========================================================
# OPENAI CLIENT
# =========================================================

client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
    timeout=4.2,
    max_retries=0
)


# =========================================================
# OPENAI STARTUP WARMUP
# =========================================================

_warmup_started = False
_warmup_lock = threading.Lock()


def warmup_openai():
    """
    Один короткий OpenAI-запрос через 3 секунды
    после запуска worker.

    Нужен для прогрева соединения Render -> OpenAI.
    Выполняется один раз на worker.
    """

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

        elapsed = time.time() - started

        print(
            f"OPENAI WARMUP OK | "
            f"TIME: {elapsed:.2f}s | "
            f"STATUS: {getattr(response, 'status', None)}",
            flush=True
        )

    except Exception as e:

        elapsed = time.time() - started

        print(
            f"OPENAI WARMUP FAILED | "
            f"TIME: {elapsed:.2f}s | "
            f"{type(e).__name__}: {e}",
            flush=True
        )


# Функция уже определена.
# Теперь запускаем фоновый warmup.
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

Не пытайся обязательно вместить весь большой ответ
в одну генерацию.

Лучше дай первую содержательную часть быстро
и закончи её нормальным предложением.

Если ответ не помещается в доступный объём,
не сокращай смысл искусственно.
Backend сможет запросить продолжение.

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
# FAST INITIAL OUTPUT BUDGET
# =========================================================

def choose_output_budget(text):
    """
    Первый ответ специально делаем небольшим.

    Цель:
    получить первую содержательную часть максимально быстро.

    Если модель упирается в max_output_tokens,
    существующий continuation-механизм догенерирует ответ.
    """

    t = normalize_command(text)

    # ---------------------------------------------
    # Сложный / длинный запрос
    # ---------------------------------------------

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

    if any(
        marker in t
        for marker in long_markers
    ):
        return 220

    # ---------------------------------------------
    # Обычное объяснение
    # ---------------------------------------------

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

    if any(
        marker in t
        for marker in medium_markers
    ):
        return 180

    # ---------------------------------------------
    # Короткий фактический вопрос
    # ---------------------------------------------

    short_markers = (
        "сколько",
        "когда",
        "кто",
        "где",
        "какая столица",
        "сколько лет"
    )

    if any(
        marker in t
        for marker in short_markers
    ):
        return 120

    # ---------------------------------------------
    # Обычный вопрос
    # ---------------------------------------------

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
        return [
            sentence.strip()
        ]

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
                current.strip()
            )

        current = part

    if current:
        chunks.append(
            current.strip()
        )

    final_chunks = []

    for chunk in chunks:

        if len(chunk) <= CHUNK_HARD_MAX:

            final_chunks.append(
                chunk
            )

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

            remaining = (
                remaining[cut:].strip()
            )

        if remaining:

            final_chunks.append(
                remaining
            )

    return final_chunks


def split_into_chunks(text):

    sentences = split_sentences(
        text
    )

    if not sentences:

        return [
            text.strip()
        ] if text.strip() else []

    normalized = []

    for sentence in sentences:

        normalized.extend(
            split_oversized_sentence(
                sentence
            )
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

                chunks.append(
                    current.strip()
                )

            current = sentence.strip()

    if current:

        chunks.append(
            current.strip()
        )

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
            "needs_model_continuation": False
        }

    return sessions[
        session_id
    ]


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

        response[
            "buttons"
        ] = [
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
# HEALTH
#
# cron-job.org вызывает только GET /
#
# OPENAI ЗДЕСЬ НЕ ВЫЗЫВАЕТСЯ.
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


        # =================================================
        # ALICE REQUEST
        # =================================================

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


        # =================================================
        # MANUAL API TEST
        # =================================================

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
        # LOCAL:
        # "ЭТО ВСЁ?"
        # =================================================

        if normalized_text in DONE_CHECK_WORDS:

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

            if (
                has_more
                or model_can_continue
            ):

                answer = (
                    "Нет, информация ещё осталась."
                )

                print(
                    "LOCAL DONE CHECK: MORE CONTENT",
                    flush=True
                )

            else:

                answer = (
                    "Да, это полный ответ "
                    "по текущему запросу."
                )

                print(
                    "LOCAL DONE CHECK: COMPLETE",
                    flush=True
                )


            if is_alice:

                return alice_response(
                    answer,
                    has_more=has_more,
                    model_can_continue=model_can_continue
                )


            return jsonify({
                "answer": answer,
                "has_more": has_more,
                "model_can_continue":
                    model_can_continue
            })


        # =================================================
        # LOCAL THANKS
        # =================================================

        if normalized_text in THANKS_WORDS:

            answer = "Пожалуйста!"

            print(
                "LOCAL THANKS",
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
        #
        # OPENAI НЕ ВЫЗЫВАЕМ.
        # =================================================

        if (
            wants_continue
            and state[
                "pending_chunks"
            ]
        ):

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


            print(
                "PENDING CHUNK SENT | "
                f"LENGTH: {len(next_chunk)} | "
                f"REMAINING: "
                f"{len(state['pending_chunks'])} | "
                f"MODEL_CONTINUE: "
                f"{model_can_continue}",
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
        # CONTINUE INCOMPLETE OPENAI RESPONSE
        # =================================================

        if (
            wants_continue
            and not state[
                "pending_chunks"
            ]
            and state[
                "needs_model_continuation"
            ]
            and state[
                "previous_response_id"
            ]
        ):

            print(
                "MODEL CONTINUATION REQUEST",
                flush=True
            )


            openai_started = (
                time.time()
            )


            try:

                response = (
                    client.responses.create(
                        model=MODEL_NAME,

                        previous_response_id=(
                            state[
                                "previous_response_id"
                            ]
                        ),

                        input=(
                            "Продолжи предыдущий ответ "
                            "точно с того места, "
                            "где он оборвался. "
                            "Не повторяй уже сказанное. "
                            "Продолжай нормальными "
                            "законченными предложениями."
                        ),

                        reasoning={
                            "effort": "none"
                        },

                        # Продолжение может быть длиннее.
                        max_output_tokens=420,

                        # Пользователь уже получил
                        # первую часть ответа.
                        timeout=5.5
                    )
                )


                full_answer = (
                    response.output_text
                    or ""
                ).strip()


                elapsed = (
                    time.time()
                    - openai_started
                )


                print(
                    "MODEL CONTINUATION TIME: "
                    f"{elapsed:.2f}s",
                    flush=True
                )


                print(
                    "CONTINUATION STATUS: "
                    f"{getattr(response, 'status', None)}",
                    flush=True
                )


                reason = (
                    get_incomplete_reason(
                        response
                    )
                )


                print(
                    "CONTINUATION INCOMPLETE REASON: "
                    f"{reason}",
                    flush=True
                )


                state[
                    "previous_response_id"
                ] = getattr(
                    response,
                    "id",
                    None
                )


                state[
                    "needs_model_continuation"
                ] = (
                    response_needs_continuation(
                        response
                    )
                )


            except APITimeoutError:

                elapsed = (
                    time.time()
                    - openai_started
                )


                print(
                    "MODEL CONTINUATION TIMEOUT after "
                    f"{elapsed:.2f}s",
                    flush=True
                )


                full_answer = (
                    "Продолжение формируется "
                    "дольше обычного. "
                    "Попробуйте ещё раз."
                )


                # Сохраняем возможность
                # повторить continuation.
                state[
                    "needs_model_continuation"
                ] = True


            except APIError as e:

                print(
                    "MODEL CONTINUATION API ERROR: "
                    f"{type(e).__name__}: {e}",
                    flush=True
                )


                full_answer = (
                    "Не удалось получить продолжение. "
                    "Попробуйте ещё раз."
                )


            except Exception as e:

                print(
                    "MODEL CONTINUATION ERROR: "
                    f"{type(e).__name__}: {e}",
                    flush=True
                )


                full_answer = (
                    "Не удалось получить продолжение. "
                    "Попробуйте ещё раз."
                )


            chunks = split_into_chunks(
                full_answer
            )


            if not chunks:

                chunks = [
                    "Не удалось сформировать продолжение."
                ]


            first_chunk = (
                chunks[0]
            )


            state[
                "pending_chunks"
            ] = chunks[1:]


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


            print(
                "CONTINUATION CHUNKS: "
                f"{len(chunks)} | "
                f"LENGTHS: "
                f"{[len(x) for x in chunks]} | "
                f"MODEL_CONTINUE: "
                f"{state['needs_model_continuation']}",
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
        # USER ASKS "CONTINUE",
        # BUT NOTHING IS LEFT
        # =================================================

        if (
            wants_continue
            and not state[
                "pending_chunks"
            ]
            and not state[
                "needs_model_continuation"
            ]
        ):

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

        if (
            text
            and not wants_continue
        ):

            if state[
                "pending_chunks"
            ]:

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


        # =================================================
        # EMPTY INPUT
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
        # QUICK LOCAL ANSWER
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
        # HISTORY
        # =================================================

        history = (
            state[
                "history"
            ][
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
        # FAST INITIAL BUDGET
        # =================================================

        output_budget = (
            choose_output_budget(
                text
            )
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
        # INITIAL OPENAI CALL
        # =================================================

        openai_started = (
            time.time()
        )


        try:

            response = (
                client.responses.create(
                    model=MODEL_NAME,

                    instructions=(
                        SYSTEM_PROMPT
                    ),

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
                "OPENAI TIME: "
                f"{openai_time:.2f}s",
                flush=True
            )


            print(
                "FULL ANSWER LENGTH: "
                f"{len(full_answer)}",
                flush=True
            )


            response_status = (
                getattr(
                    response,
                    "status",
                    None
                )
            )


            incomplete_reason = (
                get_incomplete_reason(
                    response
                )
            )


            print(
                "RESPONSE STATUS: "
                f"{response_status}",
                flush=True
            )


            print(
                "INCOMPLETE REASON: "
                f"{incomplete_reason}",
                flush=True
            )


            state[
                "previous_response_id"
            ] = getattr(
                response,
                "id",
                None
            )


            state[
                "needs_model_continuation"
            ] = (
                response_needs_continuation(
                    response
                )
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


            state[
                "previous_response_id"
            ] = None


            state[
                "needs_model_continuation"
            ] = False


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


            state[
                "previous_response_id"
            ] = None


            state[
                "needs_model_continuation"
            ] = False


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


            state[
                "previous_response_id"
            ] = None


            state[
                "needs_model_continuation"
            ] = False


        if not full_answer:

            full_answer = (
                "Не удалось сформировать ответ. "
                "Повторите вопрос."
            )


        # =================================================
        # SAVE HISTORY
        # =================================================

        history.append({
            "role": "user",
            "content": text
        })


        history.append({
            "role": "assistant",
            "content": full_answer
        })


        state[
            "history"
        ] = history[
            -MAX_HISTORY_ITEMS:
        ]


        # =================================================
        # CHUNKING
        # =================================================

        chunks = split_into_chunks(
            full_answer
        )


        if not chunks:

            chunks = [
                "Не удалось сформировать ответ."
            ]


        first_chunk = (
            chunks[0]
        )


        state[
            "pending_chunks"
        ] = chunks[1:]


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


        print(
            f"CHUNKS: {len(chunks)} | "
            f"LENGTHS: "
            f"{[len(chunk) for chunk in chunks]} | "
            f"REMAINING: "
            f"{len(state['pending_chunks'])} | "
            f"MODEL_CONTINUE: "
            f"{state['needs_model_continuation']}",
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
        # RETURN RESPONSE
        # =================================================

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
