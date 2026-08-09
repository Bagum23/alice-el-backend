import os
import time
import re
import threading
import uuid
import json
import urllib.request
import urllib.error

from flask import Flask, request, jsonify
from openai import OpenAI


app = Flask(__name__)

MODEL_NAME = "gpt-5.6-luna"


# =========================================================
# SETTINGS
# =========================================================

ALICE_WAIT_SECONDS = 3.2

BACKGROUND_OPENAI_TIMEOUT = 15.0
BACKGROUND_CONTINUATION_TIMEOUT = 15.0

EMAIL_OPENAI_TIMEOUT = 30.0

MAX_HISTORY_ITEMS = 10

CHUNK_TARGET = 700
CHUNK_HARD_MAX = 900

CONTINUE_PROMPT = "Продолжить?"


# =========================================================
# EMAIL SETTINGS
# =========================================================

USER_EMAIL = os.environ.get(
    "USER_EMAIL",
    ""
).strip()

RESEND_API_KEY = os.environ.get(
    "RESEND_API_KEY",
    ""
).strip()

RESEND_FROM_EMAIL = os.environ.get(
    "RESEND_FROM_EMAIL",
    "Пом Эл <onboarding@resend.dev>"
).strip()


# =========================================================
# OPENAI
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
# MEMORY
# =========================================================

sessions = {}


def get_session(session_id):

    if session_id not in sessions:

        sessions[session_id] = {

            "history": [],

            "pending_chunks": [],

            "previous_response_id": None,

            "needs_model_continuation": False,

            "generation_running": False,

            "generation_ready": False,

            "generation_job_id": None,

            "generation_error": None,

            # Последний содержательный ответ.
            "last_full_answer": "",

            "last_user_question": "",

            # Email.
            "email_running": False,

            # idle / sending / sent / error
            "last_email_status": "idle",

            "last_email_subject": "",

            "last_email_error": "",

            "lock": threading.Lock()
        }

    return sessions[session_id]


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

Backend самостоятельно разбивает длинные ответы
на несколько частей и умеет продолжать их.

Пиши нормальными законченными предложениями.
Старайся делать отдельное предложение короче 300 символов.

Не используй Markdown, таблицы и сложное форматирование.
Не называй себя Алисой.

Избегай воды и повторов.
"""


EMAIL_SYSTEM_PROMPT = """
Ты готовишь информационный материал для отправки по электронной почте.

Пиши по-русски.

Материал должен быть значительно подробнее обычного голосового ответа.

Структура:
краткое введение;
основная информация;
важные детали;
практические рекомендации;
краткий итог.

Используй понятные заголовки и списки там, где они полезны.

Не упоминай ограничения голосового интерфейса.

Не придумывай актуальные цены, расписания, новости
или другие данные реального времени, если они не были предоставлены.

Материал должен быть самостоятельным и понятным
без предыдущего диалога.
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


EXIT_WORDS = {

    "заканчивай",
    "заканчивай работу",
    "эл заканчивай",
    "эл заканчивай работу",
    "эл закончи работу",
    "закончи работу",
    "заверши работу",
    "завершить",
    "выход",
    "выйти",
    "стоп"
}


ALREADY_ACTIVE_WORDS = {

    "активируй пом эл",
    "активируй помощник эл",
    "запусти пом эл",
    "запусти помощник эл"
}


EMAIL_STATUS_WORDS = {

    "письмо отправилось",
    "письмо ушло",
    "отправилось письмо",
    "почта отправилась",
    "что с письмом",
    "статус письма"
}


# =========================================================
# TEXT HELPERS
# =========================================================

def normalize_command(text):

    text = (
        text or ""
    ).lower().strip()

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


# =========================================================
# EMAIL COMMAND PARSER
# =========================================================

def parse_email_request(text):

    original = (
        text or ""
    ).strip()

    normalized = normalize_command(
        original
    )

    # Это должна быть именно команда,
    # а не обычный вопрос про почту.

    if not re.match(
        r"^(?:эл\s+)?"
        r"(?:отправь|пришли|скинь|перешли)",
        normalized
    ):
        return None


    if (
        "почт" not in normalized
        and "email" not in normalized
        and "емейл" not in normalized
    ):
        return None


    # -----------------------------------------------------
    # "Отправь мне это на почту"
    # -----------------------------------------------------

    last_markers = (
        "это на почту",
        "этот ответ на почту",
        "последний ответ на почту",
        "это по почте",
        "последний ответ по почте"
    )

    if any(
        marker in normalized
        for marker in last_markers
    ):
        return {
            "type": "last"
        }


    # -----------------------------------------------------
    # "Отправь мне на почту информационный пакет по ..."
    # -----------------------------------------------------

    patterns = [

        (
            r"^(?:эл\s+)?"
            r"(?:отправь|пришли|скинь|перешли)"
            r"(?:\s+мне)?"
            r"\s+(?:на\s+почту|по\s+почте)"
            r"\s+(?:информационный\s+пакет|"
            r"пакет|информацию|материалы?)"
            r"\s+(?:по|о|об)\s+(.+)$"
        ),

        (
            r"^(?:эл\s+)?"
            r"(?:отправь|пришли|скинь|перешли)"
            r"(?:\s+мне)?"
            r"\s+(?:информационный\s+пакет|"
            r"пакет|информацию|материалы?)"
            r"\s+(?:по|о|об)\s+(.+?)"
            r"\s+(?:на\s+почту|по\s+почте)$"
        )
    ]


    for pattern in patterns:

        match = re.match(
            pattern,
            normalized
        )

        if match:

            topic = (
                match.group(1)
                .strip(" .!?")
            )

            if topic:

                return {
                    "type": "package",
                    "topic": topic
                }


    return None


# =========================================================
# QUICK ANSWERS
# =========================================================

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

        return (
            "Привет! Я Эл. "
            "Чем могу помочь?"
        )


    multiplication = re.fullmatch(
        r"\s*(-?\d+(?:[.,]\d+)?)\s*"
        r"(?:умножить\s+на|×|\*)\s*"
        r"(-?\d+(?:[.,]\d+)?)\s*",
        t
    )


    if multiplication:

        try:

            a = float(
                multiplication
                .group(1)
                .replace(",", ".")
            )

            b = float(
                multiplication
                .group(2)
                .replace(",", ".")
            )

            result = a * b

            if result.is_integer():

                result = int(
                    result
                )

            return (
                f"{multiplication.group(1)} "
                f"умножить на "
                f"{multiplication.group(2)} "
                f"равно {result}."
            )

        except Exception:

            pass


    return None


# =========================================================
# OUTPUT BUDGET
# =========================================================

def choose_output_budget(text):

    t = normalize_command(
        text
    )


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

            else
            current + " " + part
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


    return chunks


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

            else
            current + " " + sentence
        )


        if len(candidate) <= CHUNK_TARGET:

            current = candidate


        else:

            if current:

                chunks.append(
                    current.strip()
                )


            current = (
                sentence.strip()
            )


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
# ALICE RESPONSE
# =========================================================

def alice_response(
    text,
    has_more=False,
    model_can_continue=False
):

    spoken_text = (
        text.strip()
    )


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


def alice_end_response(text):

    return jsonify({

        "response": {

            "text": text,

            "tts": text,

            "end_session": True
        },

        "version": "1.0"
    })


# =========================================================
# RESEND API
# =========================================================

def resend_send_email(
    recipient,
    subject,
    body
):

    if not RESEND_API_KEY:

        raise RuntimeError(
            "RESEND_API_KEY is not configured"
        )


    if not recipient:

        raise RuntimeError(
            "USER_EMAIL is not configured"
        )


    payload = {

        "from": RESEND_FROM_EMAIL,

        "to": [
            recipient
        ],

        "subject": subject,

        "text": body
    }


    data = json.dumps(
        payload,
        ensure_ascii=False
    ).encode(
        "utf-8"
    )


    req = urllib.request.Request(

        "https://api.resend.com/emails",

        data=data,

        method="POST",

        headers={

            "Authorization":
                f"Bearer {RESEND_API_KEY}",

            "Content-Type":
                "application/json",

            # Resend требует User-Agent.
            "User-Agent":
                "alice-el-backend/1.0"
        }
    )


    try:

        with urllib.request.urlopen(
            req,
            timeout=15.0
        ) as response:

            raw = (
                response.read()
                .decode("utf-8")
            )

            return json.loads(
                raw
            )


    except urllib.error.HTTPError as e:

        try:

            body = (
                e.read()
                .decode("utf-8")
            )

        except Exception:

            body = str(e)


        raise RuntimeError(
            f"Resend HTTP {e.code}: {body}"
        )


# =========================================================
# ASYNC EMAIL SENDER
# =========================================================

def send_email_background(
    state,
    subject,
    body
):

    try:

        print(
            f"EMAIL SEND START | "
            f"TO: {USER_EMAIL} | "
            f"SUBJECT: {subject}",
            flush=True
        )


        result = resend_send_email(

            recipient=USER_EMAIL,

            subject=subject,

            body=body
        )


        email_id = (
            result.get("id")
            if isinstance(result, dict)
            else None
        )


        with state["lock"]:

            state[
                "email_running"
            ] = False

            state[
                "last_email_status"
            ] = "sent"

            state[
                "last_email_error"
            ] = ""


        print(
            f"EMAIL SENT | "
            f"ID: {email_id}",
            flush=True
        )


    except Exception as e:

        with state["lock"]:

            state[
                "email_running"
            ] = False

            state[
                "last_email_status"
            ] = "error"

            state[
                "last_email_error"
            ] = (
                f"{type(e).__name__}: {e}"
            )


        print(
            f"EMAIL ERROR | "
            f"{type(e).__name__}: {e}",
            flush=True
        )


# =========================================================
# BUILD INFORMATION PACKAGE + EMAIL
# =========================================================

def build_information_package(topic):

    full_text = ""

    response = client.responses.create(

        model=MODEL_NAME,

        instructions=EMAIL_SYSTEM_PROMPT,

        input=(
            "Подготовь полный и подробный информационный пакет "
            f"по теме: {topic}. "
            "Раскрой тему полностью. "
            "Обязательно закончи материал выводом. "
            "Не обрывай последнюю фразу."
        ),

        reasoning={
            "effort": "none"
        },

        max_output_tokens=1600,

        timeout=EMAIL_OPENAI_TIMEOUT
    )


    part = (
        response.output_text
        or ""
    ).strip()


    if part:
        full_text += part


    print(
        f"EMAIL PACKAGE PART 1 | "
        f"STATUS: {getattr(response, 'status', None)} | "
        f"REASON: {get_incomplete_reason(response)} | "
        f"LENGTH: {len(part)}",
        flush=True
    )


    # -----------------------------------------------------
    # Продолжаем автоматически, пока модель не завершит пакет
    # -----------------------------------------------------

    max_parts = 6
    part_number = 1


    while (
        response_needs_continuation(response)
        and part_number < max_parts
    ):

        previous_id = getattr(
            response,
            "id",
            None
        )


        if not previous_id:
            break


        part_number += 1


        response = client.responses.create(

            model=MODEL_NAME,

            instructions=EMAIL_SYSTEM_PROMPT,

            previous_response_id=previous_id,

            input=(
                "Продолжи информационный пакет точно с того места, "
                "где закончился предыдущий текст. "
                "Не повторяй уже написанное. "
                "Раскрой оставшиеся важные вопросы и обязательно "
                "заверши весь материал полноценным итогом. "
                "Последнее предложение должно быть законченным."
            ),

            reasoning={
                "effort": "none"
            },

            max_output_tokens=1600,

            timeout=EMAIL_OPENAI_TIMEOUT
        )


        part = (
            response.output_text
            or ""
        ).strip()


        if part:

            if full_text:
                full_text += "\n\n"

            full_text += part


        print(
            f"EMAIL PACKAGE PART {part_number} | "
            f"STATUS: {getattr(response, 'status', None)} | "
            f"REASON: {get_incomplete_reason(response)} | "
            f"LENGTH: {len(part)}",
            flush=True
        )


    # -----------------------------------------------------
    # Защита от оборванной последней фразы
    # -----------------------------------------------------

    if full_text:

        stripped = full_text.rstrip()

        if stripped[-1:] not in ".!?…":

            last_sentence_end = max(
                stripped.rfind("."),
                stripped.rfind("!"),
                stripped.rfind("?"),
                stripped.rfind("…")
            )

            if last_sentence_end > 0:
                full_text = stripped[
                    :last_sentence_end + 1
                ]


    print(
        f"EMAIL PACKAGE COMPLETE | "
        f"PARTS: {part_number} | "
        f"TOTAL LENGTH: {len(full_text)} | "
        f"FINAL STATUS: {getattr(response, 'status', None)} | "
        f"FINAL REASON: {get_incomplete_reason(response)}",
        flush=True
    )


    return full_text


def build_and_send_package(
    state,
    topic
):

    started = time.time()


    try:

        print(
            f"EMAIL PACKAGE START | "
            f"TOPIC: {topic}",
            flush=True
        )


        package = (
            build_information_package(
                topic
            )
        )


        if not package:

            raise RuntimeError(
                "Empty information package"
            )


        safe_topic = (
            topic.strip()
        )


        if len(safe_topic) > 90:

            safe_topic = (
                safe_topic[:87]
                + "..."
            )


        subject = (
            f"Пом Эл — "
            f"информационный пакет: "
            f"{safe_topic}"
        )


        result = resend_send_email(

            recipient=USER_EMAIL,

            subject=subject,

            body=package
        )


        email_id = (

            result.get("id")

            if isinstance(
                result,
                dict
            )

            else None
        )


        with state["lock"]:

            state[
                "email_running"
            ] = False

            state[
                "last_email_status"
            ] = "sent"

            state[
                "last_email_subject"
            ] = subject

            state[
                "last_email_error"
            ] = ""


        print(
            f"EMAIL PACKAGE SENT | "
            f"TIME: "
            f"{time.time() - started:.2f}s | "
            f"ID: {email_id}",
            flush=True
        )


    except Exception as e:

        with state["lock"]:

            state[
                "email_running"
            ] = False

            state[
                "last_email_status"
            ] = "error"

            state[
                "last_email_error"
            ] = (
                f"{type(e).__name__}: {e}"
            )


        print(
            f"EMAIL PACKAGE ERROR | "
            f"TIME: "
            f"{time.time() - started:.2f}s | "
            f"{type(e).__name__}: {e}",
            flush=True
        )


# =========================================================
# STORE INITIAL MODEL RESULT
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


    chunks = split_into_chunks(
        full_answer
    )


    if not chunks:

        chunks = [
            "Не удалось сформировать ответ."
        ]


    with state["lock"]:

        # Старый фоновый результат.
        if (
            state[
                "generation_job_id"
            ]
            != job_id
        ):

            print(
                f"BACKGROUND RESULT DISCARDED | "
                f"JOB: {job_id}",
                flush=True
            )

            return


        state[
            "pending_chunks"
        ] = chunks


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


        state[
            "history"
        ] = history[
            -MAX_HISTORY_ITEMS:
        ]


        state[
            "last_user_question"
        ] = user_text


        state[
            "last_full_answer"
        ] = full_answer


        state[
            "generation_running"
        ] = False


        state[
            "generation_ready"
        ] = True


        state[
            "generation_error"
        ] = None


    print(
        f"BACKGROUND READY | "
        f"JOB: {job_id} | "
        f"LENGTH: {len(full_answer)} | "
        f"STATUS: "
        f"{getattr(response, 'status', None)} | "
        f"INCOMPLETE: "
        f"{get_incomplete_reason(response)} | "
        f"CHUNKS: {[len(x) for x in chunks]}",
        flush=True
    )


# =========================================================
# BACKGROUND INITIAL GENERATION
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

            max_output_tokens=(
                output_budget
            ),

            timeout=(
                BACKGROUND_OPENAI_TIMEOUT
            )
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
                state[
                    "generation_job_id"
                ]
                == job_id
            ):

                state[
                    "generation_running"
                ] = False

                state[
                    "generation_ready"
                ] = False

                state[
                    "generation_error"
                ] = (
                    f"{type(e).__name__}: {e}"
                )


        print(
            f"BACKGROUND OPENAI ERROR | "
            f"JOB: {job_id} | "
            f"TIME: "
            f"{time.time() - started:.2f}s | "
            f"{type(e).__name__}: {e}",
            flush=True
        )


    finally:

        finished_event.set()


# =========================================================
# BACKGROUND CONTINUATION
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
                "Продолжи предыдущий ответ "
                "точно с того места, "
                "где он оборвался. "
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
                state[
                    "generation_job_id"
                ]
                != job_id
            ):

                print(
                    "BACKGROUND CONTINUATION "
                    f"DISCARDED | JOB: {job_id}",
                    flush=True
                )

                return


            state[
                "pending_chunks"
            ] = chunks


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


            # Добавляем продолжение к полному ответу.
            if full_answer:

                if state[
                    "last_full_answer"
                ]:

                    state[
                        "last_full_answer"
                    ] += (
                        "\n\n"
                        + full_answer
                    )

                else:

                    state[
                        "last_full_answer"
                    ] = full_answer


                # И обновляем историю.
                history = (
                    state[
                        "history"
                    ]
                )


                if (
                    history
                    and history[-1].get(
                        "role"
                    ) == "assistant"
                ):

                    history[-1][
                        "content"
                    ] = state[
                        "last_full_answer"
                    ]


            state[
                "generation_running"
            ] = False


            state[
                "generation_ready"
            ] = True


            state[
                "generation_error"
            ] = None


        print(
            f"BACKGROUND CONTINUATION READY | "
            f"JOB: {job_id} | "
            f"TIME: "
            f"{time.time() - started:.2f}s | "
            f"CHUNKS: {[len(x) for x in chunks]} | "
            f"MODEL_CONTINUE: "
            f"{response_needs_continuation(response)}",
            flush=True
        )


    except Exception as e:

        with state["lock"]:

            if (
                state[
                    "generation_job_id"
                ]
                == job_id
            ):

                state[
                    "generation_running"
                ] = False

                state[
                    "generation_ready"
                ] = False

                state[
                    "generation_error"
                ] = (
                    f"{type(e).__name__}: {e}"
                )


        print(
            "BACKGROUND CONTINUATION ERROR | "
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

            isinstance(
                data,
                dict
            )

            and "request" in data

            and "session" in data
        )


        # -------------------------------------------------
        # PARSE ALICE
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

                or req.get(
                    "original_utterance"
                )

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


        # =================================================
        # EXIT
        # =================================================

        if normalized_text in EXIT_WORDS:

            with state["lock"]:

                # Инвалидируем старые фоновые ответы.
                state[
                    "generation_job_id"
                ] = str(
                    uuid.uuid4()
                )

                state[
                    "pending_chunks"
                ] = []

                state[
                    "needs_model_continuation"
                ] = False


            print(
                "LOCAL EXIT",
                flush=True
            )


            if is_alice:

                return alice_end_response(
                    "Хорошо. Завершаю работу."
                )


            return jsonify({

                "answer":
                    "Хорошо. Завершаю работу.",

                "end_session": True
            })


        # =================================================
        # ALREADY ACTIVE
        # =================================================

        if normalized_text in ALREADY_ACTIVE_WORDS:

            answer = (
                "Я уже активен."
            )


            if is_alice:

                return alice_response(
                    answer
                )


            return jsonify({
                "answer": answer
            })


        # =================================================
        # EMAIL STATUS
        # =================================================

        if normalized_text in EMAIL_STATUS_WORDS:

            with state["lock"]:

                status = state[
                    "last_email_status"
                ]

                error = state[
                    "last_email_error"
                ]


            if status == "sending":

                answer = (
                    "Письмо ещё готовится."
                )


            elif status == "sent":

                answer = (
                    "Да. Письмо отправлено."
                )


            elif status == "error":

                answer = (
                    "Письмо отправить не удалось."
                )


                print(
                    f"LAST EMAIL ERROR: {error}",
                    flush=True
                )


            else:

                answer = (
                    "В этой сессии я ещё "
                    "ничего не отправлял."
                )


            if is_alice:

                return alice_response(
                    answer
                )


            return jsonify({
                "answer": answer
            })


        # =================================================
        # EMAIL COMMAND
        # =================================================

        email_request = (
            parse_email_request(
                text
            )
        )


        if email_request:

            if not USER_EMAIL:

                answer = (
                    "Адрес электронной почты "
                    "ещё не настроен."
                )


                if is_alice:

                    return alice_response(
                        answer
                    )


                return jsonify({
                    "answer": answer
                })


            if not RESEND_API_KEY:

                answer = (
                    "Сервис отправки почты "
                    "ещё не настроен."
                )


                if is_alice:

                    return alice_response(
                        answer
                    )


                return jsonify({
                    "answer": answer
                })


            with state["lock"]:

                already_sending = (
                    state[
                        "email_running"
                    ]
                )


            if already_sending:

                answer = (
                    "Предыдущее письмо "
                    "ещё готовится."
                )


                if is_alice:

                    return alice_response(
                        answer
                    )


                return jsonify({
                    "answer": answer
                })


            # -------------------------------------------------
            # SEND LAST ANSWER
            # -------------------------------------------------

            if (
                email_request[
                    "type"
                ] == "last"
            ):

                with state["lock"]:

                    last_answer = (
                        state[
                            "last_full_answer"
                        ]
                    )

                    last_question = (
                        state[
                            "last_user_question"
                        ]
                    )

                    incomplete = (
                        state[
                            "needs_model_continuation"
                        ]
                    )


                if not last_answer:

                    answer = (
                        "У меня пока нет ответа, "
                        "который можно отправить."
                    )


                elif incomplete:

                    answer = (
                        "Текущий ответ ещё не закончен. "
                        "Сначала скажите «продолжай»."
                    )


                else:

                    subject_topic = (
                        last_question
                        or "последний ответ"
                    )


                    if len(
                        subject_topic
                    ) > 90:

                        subject_topic = (
                            subject_topic[:87]
                            + "..."
                        )


                    subject = (
                        f"Пом Эл — "
                        f"{subject_topic}"
                    )


                    with state["lock"]:

                        state[
                            "email_running"
                        ] = True

                        state[
                            "last_email_status"
                        ] = "sending"

                        state[
                            "last_email_subject"
                        ] = subject

                        state[
                            "last_email_error"
                        ] = ""


                    threading.Thread(

                        target=send_email_background,

                        args=(
                            state,
                            subject,
                            last_answer
                        ),

                        name="email-last-answer",

                        daemon=True

                    ).start()


                    answer = (
                        "Принял. "
                        "Отправляю последний ответ "
                        "на вашу почту."
                    )


                if is_alice:

                    return alice_response(
                        answer
                    )


                return jsonify({
                    "answer": answer
                })


            # -------------------------------------------------
            # BUILD INFORMATION PACKAGE
            # -------------------------------------------------

            topic = (
                email_request[
                    "topic"
                ]
            )


            with state["lock"]:

                state[
                    "email_running"
                ] = True

                state[
                    "last_email_status"
                ] = "sending"

                state[
                    "last_email_subject"
                ] = topic

                state[
                    "last_email_error"
                ] = ""


            threading.Thread(

                target=build_and_send_package,

                args=(
                    state,
                    topic
                ),

                name=(
                    "email-package-"
                    + uuid.uuid4().hex[:8]
                ),

                daemon=True

            ).start()


            answer = (
                "Принял. "
                "Готовлю информационный пакет "
                "и отправляю его на вашу почту."
            )


            print(
                f"EMAIL PACKAGE QUEUED | "
                f"TOPIC: {topic}",
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
                    "Скажите «продолжай» "
                    "через несколько секунд."
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
        # THANKS
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
        # CONTINUE?
        # =================================================

        wants_continue = (

            continue_by_button

            or normalized_text
            in CONTINUE_WORDS
        )


        # =================================================
        # GENERATION STILL RUNNING
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


                if is_alice:

                    return alice_response(
                        answer
                    )


                return jsonify({
                    "answer": answer
                })


        # =================================================
        # READY CHUNK
        # =================================================

        if wants_continue:

            with state["lock"]:

                if state[
                    "pending_chunks"
                ]:

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
                    f"LENGTH: "
                    f"{len(next_chunk)} | "
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

                    "has_more":
                        has_more,

                    "model_can_continue":
                        model_can_continue
                })


        # =================================================
        # MODEL CONTINUATION
        # =================================================

        if wants_continue:

            with state["lock"]:

                needs_continuation = (
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
                needs_continuation
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
                        "continuation-"
                        + job_id[:8]
                    ),

                    daemon=True
                )


                thread.start()


                finished_event.wait(
                    ALICE_WAIT_SECONDS
                )


                with state["lock"]:

                    ready = state[
                        "generation_ready"
                    ]


                if ready:

                    with state["lock"]:

                        if state[
                            "pending_chunks"
                        ]:

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
        # NO MORE CONTENT
        # =================================================

        if wants_continue:

            answer = (
                "Это был полный ответ. "
                "Можете задать следующий вопрос."
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

            new_job_id = str(
                uuid.uuid4()
            )


            with state["lock"]:

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

            answer = (
                "Привет! Я Эл. "
                "Чем могу помочь?"
            )


            if is_alice:

                return alice_response(
                    answer
                )


            return jsonify({
                "answer": answer
            })


        # =================================================
        # QUICK LOCAL
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
        # PREPARE OPENAI REQUEST
        # =================================================

        with state["lock"]:

            history_before = list(

                state[
                    "history"
                ][
                    -MAX_HISTORY_ITEMS:
                ]
            )


        conversation = []


        for item in history_before:

            conversation.append({

                "role":
                    item["role"],

                "content":
                    item["content"]
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
                "openai-"
                + job_id[:8]
            ),

            daemon=True
        )


        thread.start()


        # =================================================
        # WAIT ONLY 3.2 SEC FOR ALICE
        # =================================================

        finished_event.wait(
            ALICE_WAIT_SECONDS
        )


        with state["lock"]:

            ready = state[
                "generation_ready"
            ]

            error = state[
                "generation_error"
            ]


        # =================================================
        # FAST RESULT
        # =================================================

        if ready:

            with state["lock"]:

                if state[
                    "pending_chunks"
                ]:

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

                print(
                    f"FAST BACKGROUND RESULT | "
                    f"TOTAL TIME: "
                    f"{time.time() - started:.2f}s",
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

                    "has_more":
                        has_more,

                    "model_can_continue":
                        model_can_continue
                })


        # =================================================
        # FAST ERROR
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
        # ALICE DEADLINE
        # =================================================

        print(
            "ALICE DEADLINE REACHED | "
            "JOB CONTINUES IN BACKGROUND | "
            f"TOTAL TIME: "
            f"{time.time() - started:.2f}s",
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
    # FATAL
    # =====================================================

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
