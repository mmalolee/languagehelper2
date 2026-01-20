import streamlit as st
from openai import OpenAI
import instructor
from pydantic import BaseModel
from io import BytesIO
from pydub import AudioSegment
from audiorecorder import audiorecorder
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
import pandas as pd
import os
import random

# Konfiguracja szerokiego układu strony i poprawka CSS dla zakładek
st.markdown(
    """
    <style>
        .block-container {
            max-width: 60%;
            padding-top: 1rem;
            padding-right: 1rem;
            padding-left: 1rem;
            padding-bottom: 1rem;
        }
        
        button[data-baseweb="tab"] {
            font-size: 14px;
            white-space: nowrap;
        }
    </style>
""",
    unsafe_allow_html=True,
)


#############
# CONSTANTS #
#############

MODEL = "gpt-4o"
TEXT_TO_SPEECH_MODEL = "tts-1"
SPEECH_TO_TEXT_MODEL = "whisper-1"
EMBEDDING_MODEL = "text-embedding-3-large"
EMBEDDING_DIM = 3072

QDRANT_COLLECTION_NAMES = ["translations", "corrections", "audio_translations"]

LANGUAGES_INPUT = sorted(["English", "German", "Italian", "Spanish", "Czech", "Polish"])
LANGUAGES_OUTPUT = sorted(
    ["English", "German", "Italian", "Spanish", "Czech", "Polish"]
)


#################
# OPENAI CLIENT #
#################


def get_openai_client():
    return OpenAI(api_key=st.session_state["OPENAI_API_KEY"])


def get_instructor_openai_client():
    return instructor.from_openai(get_openai_client())


##############
# BASEMODELS #
##############


class CheckGrammar(BaseModel):
    is_correct: bool


class Words(BaseModel):
    word: str
    translation: str
    definition: str


class Grammar(BaseModel):
    rule: str
    r_explanation: str


class Explanation(BaseModel):
    corrected: str
    difficult_words: list[Words]
    grammars: list[Grammar]


class AudioTranslationAndLanguage(BaseModel):
    translation: str
    input_language: str
    difficult_words: list[Words]


class CheckLanguage(BaseModel):
    language_detected: str


#################################
# AUDIO PROCESSING FUNCTIONS    #
#################################


def get_audio_bytes_from_file(file):
    audio_segment = AudioSegment.from_file(file)
    audio_segment_file_like = audio_segment.export(BytesIO())
    raw_bytes = audio_segment_file_like.getvalue()
    return raw_bytes


def get_audio_bytes_from_speech(speech):
    audio_segment_file_like = speech.export(BytesIO())
    raw_bytes = audio_segment_file_like.getvalue()
    return raw_bytes


##########
# LOGIC  #
##########


def translate_text(text, language_input, language_output):
    openai_client = get_openai_client()
    translation = openai_client.chat.completions.create(
        model=MODEL,
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": f"""You are a professional translator that translates from {language_input} to {language_output}.
                Preserve formatting, punctuation, and capitalization exactly as in the input. Translate word by word if necessary. Output only the translated text.""",
            },
            {"role": "user", "content": text.strip()},
        ],
    )
    return translation.choices[0].message.content


def check_language(to_check):
    instructor_openai_client = get_instructor_openai_client()
    response = instructor_openai_client.chat.completions.create(
        model=MODEL,
        temperature=0,
        response_model=CheckLanguage,
        messages=[
            {
                "role": "system",
                "content": f"""You are a language detector. Detect the language of the "{to_check}". Return detected language, e.g. "Polish". Do not return dialects, variations, or pidgin forms.
                Do not guess meanings of words in other languages.""",
            },
            {"role": "user", "content": to_check},
        ],
    )
    return response.language_detected


def check_grammar(to_check, input_language):
    instructor_openai_client = get_instructor_openai_client()
    response = instructor_openai_client.chat.completions.create(
        model=MODEL,
        temperature=0,
        response_model=CheckGrammar,
        messages=[
            {
                "role": "system",
                "content": f"You are a professional grammar specialist in {input_language}. Evaluate the grammar of {to_check}.",
            },
            {"role": "user", "content": to_check},
        ],
    )
    return response.is_correct


def get_corr_words_and_grammar(to_correct, input_language):
    instructor_openai_client = get_instructor_openai_client()
    explanation = instructor_openai_client.chat.completions.create(
        model=MODEL,
        temperature=0,
        response_model=Explanation,
        messages=[
            {
                "role": "system",
                "content": f"""You are a professional grammar specialist in {input_language}. Do 4 things:
                    1. Correct the grammar of the input text.
                    2. Extract difficult vocabulary, phrasal verbs, or idioms from the input text.
                       - 'word': The specific phrase or collocation in {input_language} (e.g., 'look at', 'take off').
                       - 'translation': The direct, short translation in English (e.g. 'patrzeć na', 'startować'). This is for flashcards.
                       - 'definition': A short explanation or definition in English to provide context.
                    3. Explain any difficult or tricky grammatical constructions found in the original text.
                    4. All explanations must be given in English.
                    """,
            },
            {"role": "user", "content": to_correct.strip()},
        ],
    ).model_dump()

    corrected_form = explanation["corrected"]
    difficult_words = explanation["difficult_words"]
    grammar = explanation["grammars"]

    return corrected_form, difficult_words, grammar


def text_to_speech(text):
    openai_client = get_openai_client()
    speech = openai_client.audio.speech.create(
        model=TEXT_TO_SPEECH_MODEL, voice="onyx", response_format="mp3", input=text
    )
    return speech.content


def speech_to_text(speech):
    openai_client = get_openai_client()
    speech_file_like = BytesIO(speech)
    speech_file_like.name = "audio.mp3"
    transcription = openai_client.audio.transcriptions.create(
        model=SPEECH_TO_TEXT_MODEL,
        file=speech_file_like,
        response_format="verbose_json",
    )
    return transcription.text


def translate_transcription(transcription, output_language):
    instructor_openai_client = get_instructor_openai_client()
    translation_obj = instructor_openai_client.chat.completions.create(
        model=MODEL,
        temperature=0,
        response_model=AudioTranslationAndLanguage,
        messages=[
            {
                "role": "system",
                "content": f"""You are a professional personal {output_language} translator. Do three things:
                1. Detect the language from "{transcription}" and translate to {output_language}. Preserve formatting.
                2. Return full detected language name (e.g. Polish, German).
                3. Extract difficult vocabulary, phrasal verbs, or idioms from the SOURCE text.
                   - For 'word': provide the specific source phrase.
                   - For 'translation': provide ONLY the direct translation in {output_language}.
                   - For 'definition': provide a short definition/context in {output_language}.""",
            },
            {"role": "user", "content": transcription.strip()},
        ],
    )
    return (
        translation_obj.translation,
        translation_obj.input_language,
        translation_obj.difficult_words,
    )


def get_embedding(text):
    openai_client = get_openai_client()
    result = openai_client.embeddings.create(
        input=[text], model=EMBEDDING_MODEL, dimensions=EMBEDDING_DIM
    )
    return result.data[0].embedding


##########
# QDRANT #
##########


@st.cache_resource
def get_qdrant_client():
    return QdrantClient(":memory:")


def assure_qdrant_collections_exist():
    qdrant_client = get_qdrant_client()

    for name in QDRANT_COLLECTION_NAMES:
        if not qdrant_client.collection_exists(name):
            if name == QDRANT_COLLECTION_NAMES[0]:
                config = {
                    "Input_Language_Vector": VectorParams(
                        size=EMBEDDING_DIM, distance=Distance.COSINE
                    ),
                    "Input_Vector": VectorParams(
                        size=EMBEDDING_DIM, distance=Distance.COSINE
                    ),
                    "Translation_Language_Vector": VectorParams(
                        size=EMBEDDING_DIM, distance=Distance.COSINE
                    ),
                    "Translation_Vector": VectorParams(
                        size=EMBEDDING_DIM, distance=Distance.COSINE
                    ),
                }
            elif name == QDRANT_COLLECTION_NAMES[1]:
                config = {
                    "Input_Language_Vector": VectorParams(
                        size=EMBEDDING_DIM, distance=Distance.COSINE
                    ),
                    "Input_Text_Vector": VectorParams(
                        size=EMBEDDING_DIM, distance=Distance.COSINE
                    ),
                    "Correction_Vector": VectorParams(
                        size=EMBEDDING_DIM, distance=Distance.COSINE
                    ),
                    "Diff_Words_Vector": VectorParams(
                        size=EMBEDDING_DIM, distance=Distance.COSINE
                    ),
                    "Grammar_Rules_Vector": VectorParams(
                        size=EMBEDDING_DIM, distance=Distance.COSINE
                    ),
                }
            else:
                config = {
                    "Input_Language_Vector": VectorParams(
                        size=EMBEDDING_DIM, distance=Distance.COSINE
                    ),
                    "Transcription_Vector": VectorParams(
                        size=EMBEDDING_DIM, distance=Distance.COSINE
                    ),
                    "Translation_Language_Vector": VectorParams(
                        size=EMBEDDING_DIM, distance=Distance.COSINE
                    ),
                    "Translation_Vector": VectorParams(
                        size=EMBEDDING_DIM, distance=Distance.COSINE
                    ),
                }
            qdrant_client.create_collection(collection_name=name, vectors_config=config)


def add_translation_to_db(input_language, input, translation_language, translation):
    qdrant_client = get_qdrant_client()
    points_count = qdrant_client.count(
        collection_name=QDRANT_COLLECTION_NAMES[0], exact=True
    )
    qdrant_client.upsert(
        collection_name=QDRANT_COLLECTION_NAMES[0],
        points=[
            PointStruct(
                id=points_count.count + 1,
                vector={
                    "Input_Language_Vector": get_embedding(input_language),
                    "Input_Vector": get_embedding(input),
                    "Translation_Language_Vector": get_embedding(translation_language),
                    "Translation_Vector": get_embedding(translation),
                },
                payload={
                    "Input_Text_Language": input_language,
                    "Input_Text": input,
                    "Translation_Language": translation_language,
                    "Translation": translation,
                },
            )
        ],
    )


def add_vocabulary_to_db(input_language, words_list, translation_language="English"):
    qdrant_client = get_qdrant_client()
    for item in words_list:
        points_count = qdrant_client.count(
            collection_name=QDRANT_COLLECTION_NAMES[0], exact=True
        )
        if isinstance(item, dict):
            word = item.get("word", "").strip()
            translation = item.get("translation", "").strip()
        else:
            word = item.word.strip()
            translation = item.translation.strip()

        qdrant_client.upsert(
            collection_name=QDRANT_COLLECTION_NAMES[0],
            points=[
                PointStruct(
                    id=points_count.count + 1,
                    vector={
                        "Input_Language_Vector": get_embedding(input_language),
                        "Input_Vector": get_embedding(word),
                        "Translation_Language_Vector": get_embedding(
                            translation_language
                        ),
                        "Translation_Vector": get_embedding(translation),
                    },
                    payload={
                        "Input_Text_Language": input_language,
                        "Input_Text": word,
                        "Translation_Language": translation_language,
                        "Translation": translation,
                    },
                )
            ],
        )


def add_correction_to_db(input_language, input, correction, diff_words, grammar_rules):
    qdrant_client = get_qdrant_client()
    points_count = qdrant_client.count(
        collection_name=QDRANT_COLLECTION_NAMES[1], exact=True
    )
    qdrant_client.upsert(
        collection_name=QDRANT_COLLECTION_NAMES[1],
        points=[
            PointStruct(
                id=points_count.count + 1,
                vector={
                    "Input_Language_Vector": get_embedding(input_language),
                    "Input_Text_Vector": get_embedding(input),
                    "Correction_Vector": get_embedding(correction),
                    "Diff_Words_Vector": get_embedding(diff_words),
                    "Grammar_Rules_Vector": get_embedding(grammar_rules),
                },
                payload={
                    "Input_Language": input_language,
                    "Input_Text": input,
                    "Correction": correction,
                    "Diff_Words": diff_words,
                    "Grammar_Rules": grammar_rules,
                },
            )
        ],
    )


def add_audio_translation_to_db(
    input_language, transcription, translation_language, translation
):
    qdrant_client = get_qdrant_client()
    points_count = qdrant_client.count(
        collection_name=QDRANT_COLLECTION_NAMES[2], exact=True
    )
    qdrant_client.upsert(
        collection_name=QDRANT_COLLECTION_NAMES[2],
        points=[
            PointStruct(
                id=points_count.count + 1,
                vector={
                    "Input_Language_Vector": get_embedding(input_language),
                    "Transcription_Vector": get_embedding(transcription),
                    "Translation_Language_Vector": get_embedding(translation_language),
                    "Translation_Vector": get_embedding(translation),
                },
                payload={
                    "Input_Language": input_language,
                    "Transcription": transcription,
                    "Translation_Language": translation_language,
                    "Translation": translation,
                },
            )
        ],
    )


def list_translations(query=None):
    qdrant_client = get_qdrant_client()
    if not query:
        translations_from_db = qdrant_client.scroll(
            collection_name=QDRANT_COLLECTION_NAMES[0], limit=10
        )[0]
        results = []
        for item in translations_from_db:
            results.append(
                {
                    "Input_Language": item.payload.get("Input_Text_Language"),
                    "Input_Text": item.payload.get("Input_Text"),
                    "Target_Language": item.payload.get("Translation_Language"),
                    "Translation": item.payload.get("Translation"),
                }
            )
        return pd.DataFrame(results)
    else:
        search_result = qdrant_client.search(
            collection_name=QDRANT_COLLECTION_NAMES[0],
            query_vector=("Input_Vector", get_embedding(query)),
            limit=10,
        )
        results = []
        for item in search_result:
            results.append(
                {
                    "Similarity": round(item.score, 2),
                    "Input_Language": item.payload.get("Input_Text_Language"),
                    "Input_Text": item.payload.get("Input_Text"),
                    "Target_Language": item.payload.get("Translation_Language"),
                    "Translation": item.payload.get("Translation"),
                }
            )
        return pd.DataFrame(results)


def list_corrections(query=None):
    qdrant_client = get_qdrant_client()
    if not query:
        corrections_from_db = qdrant_client.scroll(
            collection_name=QDRANT_COLLECTION_NAMES[1], limit=10
        )[0]
        results = []
        for item in corrections_from_db:
            results.append(
                {
                    "Input_Language": item.payload.get("Input_Language"),
                    "Input_Text": item.payload.get("Input_Text"),
                    "Correction": item.payload.get("Correction"),
                    "Difficult_Words": item.payload.get("Diff_Words"),
                    "Grammar_Rules": item.payload.get("Grammar_Rules"),
                }
            )
        return pd.DataFrame(results)
    else:
        search_result = qdrant_client.search(
            collection_name=QDRANT_COLLECTION_NAMES[1],
            query_vector=("Input_Text_Vector", get_embedding(query)),
            limit=10,
        )
        results = []
        for item in search_result:
            results.append(
                {
                    "Similarity": round(item.score, 2),
                    "Input_Language": item.payload.get("Input_Language"),
                    "Input_Text": item.payload.get("Input_Text"),
                    "Correction": item.payload.get("Correction"),
                    "Difficult_Words": item.payload.get("Diff_Words"),
                    "Grammar_Rules": item.payload.get("Grammar_Rules"),
                }
            )
        return pd.DataFrame(results)


def list_audio_translations(query=None):
    qdrant_client = get_qdrant_client()
    if not query:
        audio_from_db = qdrant_client.scroll(
            collection_name=QDRANT_COLLECTION_NAMES[2], limit=10
        )[0]
        results = []
        for item in audio_from_db:
            results.append(
                {
                    "Input_Language": item.payload.get("Input_Language"),
                    "Transcription": item.payload.get("Transcription"),
                    "Target_Language": item.payload.get("Translation_Language"),
                    "Translation": item.payload.get("Translation"),
                }
            )
        return pd.DataFrame(results)
    else:
        search_result = qdrant_client.search(
            collection_name=QDRANT_COLLECTION_NAMES[2],
            query_vector=("Transcription_Vector", get_embedding(query)),
            limit=10,
        )
        results = []
        for item in search_result:
            results.append(
                {
                    "Similarity": round(item.score, 2),
                    "Input_Language": item.payload.get("Input_Language"),
                    "Transcription": item.payload.get("Transcription"),
                    "Target_Language": item.payload.get("Translation_Language"),
                    "Translation": item.payload.get("Translation"),
                }
            )
        return pd.DataFrame(results)


def get_quiz_data(input_lang, output_lang):
    qdrant_client = get_qdrant_client()
    points, _ = qdrant_client.scroll(
        collection_name=QDRANT_COLLECTION_NAMES[0], limit=1000, with_payload=True
    )
    candidates = []
    for p in points:
        payload = p.payload
        text = payload.get("Input_Text", "")
        if len(text.split()) <= 4:
            if (
                payload.get("Input_Text_Language") == input_lang
                and payload.get("Translation_Language") == output_lang
            ):
                candidates.append(
                    {"question": text, "answer": payload.get("Translation")}
                )
    return candidates


##################
# SESSION STATES #
##################

# --- Text Translation ---
if "TT_input_language" not in st.session_state:
    st.session_state["TT_input_language"] = ""
if "TT_output_language" not in st.session_state:
    st.session_state["TT_output_language"] = ""
if "TT_detected_language" not in st.session_state:
    st.session_state["TT_detected_language"] = ""
if "TT_translate_text_input" not in st.session_state:
    st.session_state["TT_translate_text_input"] = ""
if "TT_translate_text_output" not in st.session_state:
    st.session_state["TT_translate_text_output"] = ""
if "TT_audio_in" not in st.session_state:
    st.session_state["TT_audio_in"] = None
if "TT_audio_out" not in st.session_state:
    st.session_state["TT_audio_out"] = None

# --- Audio Translation ---
if "TA_output_language" not in st.session_state:
    st.session_state["TA_output_language"] = ""
if "TA_input_language" not in st.session_state:
    st.session_state["TA_input_language"] = ""
if "TA_input_transcription" not in st.session_state:
    st.session_state["TA_input_transcription"] = ""
if "TA_output_text" not in st.session_state:
    st.session_state["TA_output_text"] = ""
if "TA_audio_out" not in st.session_state:
    st.session_state["TA_audio_out"] = None

# --- Text Correction ---
if "CT_input_language" not in st.session_state:
    st.session_state["CT_input_language"] = ""
if "CT_input_text" not in st.session_state:
    st.session_state["CT_input_text"] = ""
if "CT_output_text" not in st.session_state:
    st.session_state["CT_output_text"] = ""
if "CT_diff_words" not in st.session_state:
    st.session_state["CT_diff_words"] = ""
if "CT_grammar_rules" not in st.session_state:
    st.session_state["CT_grammar_rules"] = ""
if "CT_audio_out" not in st.session_state:
    st.session_state["CT_audio_out"] = None

# --- Speech Correction ---
if "CS_input_language" not in st.session_state:
    st.session_state["CS_input_language"] = ""
if "CS_input_speech_as_text" not in st.session_state:
    st.session_state["CS_input_speech_as_text"] = ""
if "CS_output_as_text" not in st.session_state:
    st.session_state["CS_output_as_text"] = ""
if "CS_diff_words" not in st.session_state:
    st.session_state["CS_diff_words"] = ""
if "CS_grammar_rules" not in st.session_state:
    st.session_state["CS_grammar_rules"] = ""
if "CS_audio_out" not in st.session_state:
    st.session_state["CS_audio_out"] = None

# --- Quiz ---
if "quiz_active" not in st.session_state:
    st.session_state["quiz_active"] = False
if "quiz_questions" not in st.session_state:
    st.session_state["quiz_questions"] = []
if "quiz_current_index" not in st.session_state:
    st.session_state["quiz_current_index"] = 0
if "quiz_score" not in st.session_state:
    st.session_state["quiz_score"] = 0
if "quiz_submitted" not in st.session_state:
    st.session_state["quiz_submitted"] = False
if "quiz_finished_final" not in st.session_state:
    st.session_state["quiz_finished_final"] = False
if "quiz_mode" not in st.session_state:
    st.session_state["quiz_mode"] = "Written Answer"

# --- API Key ---
if "OPENAI_API_KEY" not in st.session_state:
    st.session_state["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "")

# Ask for API Key if not present
if not st.session_state.get("OPENAI_API_KEY"):
    st.info("Enter your OpenAI API key")
    st.session_state["OPENAI_API_KEY"] = st.text_input(
        "OpenAI API Key", type="password"
    )
    if st.session_state["OPENAI_API_KEY"]:
        st.rerun()
    st.stop()


########
# MAIN #
########

st.title("Language Helper")

assure_qdrant_collections_exist()

(
    text_tr,
    audio_tr,
    text_cor,
    speech_cor,
    quiz_tab,
    search_translation,
    search_audio,
    search_correction,
) = st.tabs(
    [
        "Translate Text",
        "Translate Audio",
        "Correct Text",
        "Record and Correct Speech",
        "Vocabulary Quiz",
        "Search Translation",
        "Search Audio Translations",
        "Search Correction",
    ]
)


# 1. Text Translation
with text_tr:
    c0, c1 = st.columns(2)
    translate_button = st.button("Translate Text", use_container_width=True)
    save_button = st.button("Save Translation", use_container_width=True)

    with c0:
        st.session_state["TT_input_language"] = st.selectbox(
            "Select Input Language",
            LANGUAGES_INPUT,
            help="Choose the language you want translate from",
        )
        st.session_state["TT_translate_text_input"] = st.text_area(
            "Enter the Text",
            height=150,
            value=st.session_state["TT_translate_text_input"],
        ).strip()

    with c1:
        st.session_state["TT_output_language"] = st.selectbox(
            "Select Output Language",
            LANGUAGES_OUTPUT,
            help="Choose the language you want translate to",
        )

    if translate_button:
        if st.session_state["TT_translate_text_input"]:
            st.session_state["TT_detected_language"] = check_language(
                st.session_state["TT_translate_text_input"]
            )
            if (
                st.session_state["TT_detected_language"]
                == st.session_state["TT_input_language"]
            ):

                st.session_state["TT_audio_in"] = text_to_speech(
                    st.session_state["TT_translate_text_input"]
                )

                st.session_state["TT_translate_text_output"] = translate_text(
                    st.session_state["TT_translate_text_input"],
                    st.session_state["TT_input_language"],
                    st.session_state["TT_output_language"],
                )

                st.session_state["TT_audio_out"] = text_to_speech(
                    st.session_state["TT_translate_text_output"]
                )
                st.toast("Text Has Been Translated!")
            else:
                st.warning(
                    f"Input and Detected Language Don't Match. Switch Language Input to {st.session_state['TT_detected_language']}"
                )
        else:
            st.warning("Enter the Text")

    # WIDOK (renderowany zawsze)
    if st.session_state.get("TT_translate_text_output"):
        with c0:
            if st.session_state.get("TT_audio_in"):
                st.audio(st.session_state["TT_audio_in"], format="audio/mpeg")
        with c1:
            st.text_area(
                "Translation",
                height=150,
                value=st.session_state["TT_translate_text_output"],
            )
            if st.session_state.get("TT_audio_out"):
                st.audio(st.session_state["TT_audio_out"], format="audio/mpeg")

    if save_button:
        if st.session_state["TT_translate_text_output"]:
            add_translation_to_db(
                st.session_state["TT_input_language"],
                st.session_state["TT_translate_text_input"],
                st.session_state["TT_output_language"],
                st.session_state["TT_translate_text_output"],
            )
            st.toast("Translation Has Been Saved!")
            st.session_state["TT_translate_text_output"] = ""
            st.session_state["TT_audio_in"] = None
            st.session_state["TT_audio_out"] = None
        else:
            st.warning("No Translation to Save")


# 2. Audio Translation
with audio_tr:
    st.session_state["TA_output_language"] = st.selectbox(
        "Select Desired Output Language", LANGUAGES_OUTPUT
    )
    uploaded_file = st.file_uploader("Upload a file", type=["mp3", "wav"])

    if uploaded_file:
        raw_audio_bytes = get_audio_bytes_from_file(uploaded_file)
        st.audio(raw_audio_bytes)

        translate_audio_button = st.button(
            "Translate Audio Content", use_container_width=True
        )
        save_button = st.button("Save Audio Translation", use_container_width=True)

        if translate_audio_button:
            st.session_state["TA_input_transcription"] = speech_to_text(raw_audio_bytes)
            (
                st.session_state["TA_output_text"],
                st.session_state["TA_input_language"],
                difficult_words_audio,
            ) = translate_transcription(
                st.session_state["TA_input_transcription"],
                st.session_state["TA_output_language"],
            )

            st.session_state["TA_raw_diff_words"] = difficult_words_audio
            st.session_state["TA_audio_out"] = text_to_speech(
                st.session_state["TA_output_text"]
            )

        # WIDOK
        if st.session_state.get("TA_output_text"):
            with st.container(border=True):
                st.header(
                    f"Provided audio is in {st.session_state['TA_input_language']}"
                )
                st.text_area(
                    "Audio Translation", value=st.session_state["TA_output_text"]
                )

                if st.session_state.get("TA_raw_diff_words"):
                    with st.expander("Extracted Vocabulary"):
                        for w in st.session_state["TA_raw_diff_words"]:
                            st.write(f"**{w.word}**: {w.translation} ({w.definition})")

                if st.session_state.get("TA_audio_out"):
                    st.audio(st.session_state["TA_audio_out"], format="audio/mpeg")

        if save_button:
            if st.session_state["TA_output_text"]:
                add_audio_translation_to_db(
                    st.session_state["TA_input_language"],
                    st.session_state["TA_input_transcription"],
                    st.session_state["TA_output_language"],
                    st.session_state["TA_output_text"],
                )

                if (
                    "TA_raw_diff_words" in st.session_state
                    and st.session_state["TA_raw_diff_words"]
                ):
                    add_vocabulary_to_db(
                        st.session_state["TA_input_language"],
                        st.session_state["TA_raw_diff_words"],
                        st.session_state["TA_output_language"],
                    )
                    st.toast("Vocabulary added to Quiz!")

                st.toast("Audio Translation Has Been Saved!")
                st.session_state["TA_output_text"] = ""
                st.session_state["TA_audio_out"] = None
            else:
                st.warning("No Translation to Save")


# 3. Text Correction
with text_cor:
    st.session_state["CT_input_language"] = st.selectbox(
        "Select Language", LANGUAGES_INPUT, key="ct_lang"
    )
    st.session_state["CT_input_text"] = st.text_area(
        "Enter the Sentence", height=75
    ).strip()

    correct_button = st.button("Fix and Explain", use_container_width=True)
    save_button = st.button("Save Correction", use_container_width=True)

    if correct_button:
        if st.session_state["CT_input_text"]:
            detected_language = check_language(st.session_state["CT_input_text"])
            if st.session_state["CT_input_language"] == detected_language:
                if check_grammar(
                    st.session_state["CT_input_text"],
                    st.session_state["CT_input_language"],
                ):
                    st.success("The Sentence is Correct")
                else:
                    (st.session_state["CT_output_text"], diff_words, grammars) = (
                        get_corr_words_and_grammar(
                            st.session_state["CT_input_text"],
                            st.session_state["CT_input_language"],
                        )
                    )

                    st.session_state["CT_raw_diff_words"] = diff_words

                    words_expl = (
                        "".join(
                            [
                                f"'{w['word'].capitalize()}' : {w['translation']} ({w['definition']})\n"
                                for w in diff_words
                            ]
                        )
                        or "No tricky words."
                    )
                    gram_expl = (
                        "".join(
                            [
                                f"'{g['rule']}' : {g['r_explanation']}\n"
                                for g in grammars
                            ]
                        )
                        or "No tricky rules."
                    )

                    st.session_state["CT_diff_words"] = words_expl
                    st.session_state["CT_grammar_rules"] = gram_expl

                    st.session_state["CT_audio_out"] = text_to_speech(
                        st.session_state["CT_output_text"]
                    )
                    st.toast("Content Has Been Corrected!")
            else:
                st.warning(
                    f"Input and Detected Language Don't Match. Switch to {detected_language}"
                )
        else:
            st.warning("Enter the Content")

    # WIDOK
    if st.session_state.get("CT_output_text"):
        with st.container(border=True):
            st.header("Correct Form")
            st.text_area("", value=st.session_state["CT_output_text"])
            with st.expander("Tricky Words"):
                st.text_area("", value=st.session_state["CT_diff_words"])
            with st.expander("Grammar"):
                st.text_area("", value=st.session_state["CT_grammar_rules"])
            with st.expander("Pronunciation"):
                if st.session_state.get("CT_audio_out"):
                    st.audio(st.session_state["CT_audio_out"], format="audio/mpeg")

    if save_button:
        if st.session_state["CT_output_text"]:
            add_correction_to_db(
                st.session_state["CT_input_language"],
                st.session_state["CT_input_text"],
                st.session_state["CT_output_text"],
                st.session_state["CT_diff_words"],
                st.session_state["CT_grammar_rules"],
            )

            if (
                "CT_raw_diff_words" in st.session_state
                and st.session_state["CT_raw_diff_words"]
            ):
                add_vocabulary_to_db(
                    st.session_state["CT_input_language"],
                    st.session_state["CT_raw_diff_words"],
                    "English",
                )
                st.toast("Vocabulary from correction added to Quiz!")

            st.toast("Correction Has Been Saved!")
            st.session_state["CT_output_text"] = ""
            st.session_state["CT_audio_out"] = None
        else:
            st.warning("No Correction to Save")


# 4. Record and Correct Speech
with speech_cor:
    check_audio_correction_button = st.button("Check and Fix", use_container_width=True)
    save_button = st.button("Save Speech Correction", use_container_width=True)

    st.session_state["CS_input_language"] = st.selectbox(
        "Select Recorded Speech Language", LANGUAGES_INPUT, help="Select input language"
    )

    c0, c1 = st.columns([0.32, 0.68])
    with c0:
        input_speech = audiorecorder(
            start_prompt="Record Sentence", stop_prompt="Stop Recording Sentence"
        )

    if not input_speech and check_audio_correction_button:
        st.warning("Enter the Content")

    if input_speech:
        with c1:
            input_speech_bytes = get_audio_bytes_from_speech(input_speech)
            transcription = speech_to_text(input_speech_bytes)
            st.audio(input_speech_bytes)

        st.session_state["CS_input_speech_as_text"] = st.text_area(
            "Check and Correct the Recording", value=transcription
        )

        language_detected = check_language(st.session_state["CS_input_speech_as_text"])

        if check_audio_correction_button:
            if language_detected == st.session_state["CS_input_language"]:
                if check_grammar(
                    st.session_state["CS_input_speech_as_text"],
                    st.session_state["CS_input_language"],
                ):
                    st.success("Sentence is Correct!")
                else:
                    sentence, diff_words, grammars = get_corr_words_and_grammar(
                        st.session_state["CS_input_speech_as_text"],
                        st.session_state["CS_input_language"],
                    )

                    st.session_state["CS_raw_diff_words"] = diff_words

                    words_expl = (
                        "".join(
                            [
                                f"'{w['word'].capitalize()}' : {w['translation']} ({w['definition']})\n"
                                for w in diff_words
                            ]
                        )
                        or "No tricky words."
                    )
                    gram_expl = (
                        "".join(
                            [
                                f"'{g['rule']}' : {g['r_explanation']}\n"
                                for g in grammars
                            ]
                        )
                        or "No tricky rules."
                    )

                    st.session_state["CS_output_as_text"] = sentence
                    st.session_state["CS_diff_words"] = words_expl
                    st.session_state["CS_grammar_rules"] = gram_expl

                    st.session_state["CS_audio_out"] = text_to_speech(
                        st.session_state["CS_output_as_text"]
                    )

            else:
                st.warning(
                    f"Input and Detected Language Don't Match. Switch to {language_detected}"
                )

    # WIDOK
    if st.session_state.get("CS_output_as_text"):
        with st.container(border=True):
            st.header("Correct Form")
            st.text_area("", value=st.session_state["CS_output_as_text"])
            with st.expander("Tricky Words"):
                st.text_area("", value=st.session_state["CS_diff_words"])
            with st.expander("Grammar"):
                st.text_area("", value=st.session_state["CS_grammar_rules"])
            with st.expander("Pronunciation"):
                if st.session_state.get("CS_audio_out"):
                    st.audio(st.session_state["CS_audio_out"], format="audio/mpeg")

    if save_button:
        if st.session_state.get("CS_output_as_text"):
            add_correction_to_db(
                st.session_state["CS_input_language"],
                st.session_state["CS_input_speech_as_text"],
                st.session_state["CS_output_as_text"],
                st.session_state["CS_diff_words"],
                st.session_state["CS_grammar_rules"],
            )

            if "CS_raw_diff_words" in st.session_state:
                add_vocabulary_to_db(
                    st.session_state["CS_input_language"],
                    st.session_state["CS_raw_diff_words"],
                    "English",
                )
                st.toast("Difficult words added to your Quiz database!")

            st.toast("Correction saved!")
            st.session_state["CS_output_as_text"] = ""
            st.session_state["CS_input_speech_as_text"] = ""
            st.session_state["CS_audio_out"] = None
        else:
            st.warning("No Correction to Save")


# 5. Quiz Tab
with quiz_tab:
    st.header("Vocabulary & Phrase Quiz")

    if not st.session_state.get("quiz_active", False) and not st.session_state.get(
        "quiz_finished_final", False
    ):
        st.subheader("Quiz Settings")
        c1, c2 = st.columns(2)
        with c1:
            q_src = st.selectbox("From Language", LANGUAGES_INPUT, key="q_lang_in")
        with c2:
            q_tgt = st.selectbox("To Language", LANGUAGES_OUTPUT, key="q_lang_out")

        quiz_mode = st.radio(
            "Select Quiz Mode",
            ["Written Answer", "Select Translation"],
            horizontal=True,
        )

        if q_src == q_tgt:
            st.error("Source and Target languages must be different.")
        else:
            candidates = get_quiz_data(q_src, q_tgt)
            count = len(candidates)
            if count < 5:
                st.warning(
                    f"Not enough items ({count}). You need at least 5 short phrases/words."
                )
            else:
                st.success(f"Found {count} items.")
                num_q = st.slider(
                    "Select number of questions", 1, min(count, 20), min(count, 5)
                )

                if st.button("Start Quiz", use_container_width=True):
                    selected = random.sample(candidates, num_q)

                    if quiz_mode == "Select Translation":
                        for item in selected:
                            other = [
                                c["answer"]
                                for c in candidates
                                if c["answer"] != item["answer"]
                            ]
                            distractors = random.sample(other, min(3, len(other)))
                            options = distractors + [item["answer"]]
                            random.shuffle(options)
                            item["options"] = options

                    st.session_state["quiz_questions"] = selected
                    st.session_state["quiz_mode"] = quiz_mode
                    st.session_state["quiz_current_index"] = 0
                    st.session_state["quiz_score"] = 0
                    st.session_state["quiz_active"] = True
                    st.session_state["quiz_submitted"] = False
                    st.rerun()

    elif st.session_state.get("quiz_active", False):
        idx = st.session_state["quiz_current_index"]
        questions = st.session_state["quiz_questions"]
        current = questions[idx]
        mode = st.session_state["quiz_mode"]

        st.write(f"Question {idx + 1} of {len(questions)}")
        st.progress((idx) / len(questions))

        with st.container(border=True):
            st.subheader(f"How do you translate: '{current['question']}'?")
            if st.button("Listen to Question", key=f"q_audio_{idx}"):
                with st.spinner("Generating..."):
                    st.audio(text_to_speech(current["question"]), format="audio/mpeg")

            if mode == "Written Answer":
                user_ans = st.text_input(
                    "Your answer:",
                    key=f"input_q_{idx}",
                    disabled=st.session_state["quiz_submitted"],
                )
            else:
                user_ans = st.radio(
                    "Choose correct translation:",
                    current["options"],
                    key=f"radio_q_{idx}",
                    disabled=st.session_state["quiz_submitted"],
                )

            if not st.session_state["quiz_submitted"]:
                if st.button("Check Answer", use_container_width=True):
                    st.session_state["quiz_submitted"] = True
                    st.rerun()
            else:
                is_correct = (
                    user_ans.strip().lower() == current["answer"].strip().lower()
                )
                if is_correct:
                    st.success(f"Correct. Answer: {current['answer']}")
                    if f"scored_{idx}" not in st.session_state:
                        st.session_state["quiz_score"] += 1
                        st.session_state[f"scored_{idx}"] = True
                else:
                    st.error(f"Incorrect. Answer: {current['answer']}")

                if st.button("Listen to Answer", key=f"ans_audio_{idx}"):
                    st.audio(text_to_speech(current["answer"]), format="audio/mpeg")

                if idx + 1 < len(questions):
                    if st.button("Next Question", use_container_width=True):
                        st.session_state["quiz_current_index"] += 1
                        st.session_state["quiz_submitted"] = False
                        st.rerun()
                else:
                    if st.button("Finish Quiz", use_container_width=True):
                        st.session_state["quiz_active"] = False
                        st.session_state["quiz_finished_final"] = True
                        st.rerun()

    elif st.session_state.get("quiz_finished_final", False):
        st.header("Quiz Results")
        st.metric(
            "Final Score",
            f"{st.session_state['quiz_score']} / {len(st.session_state['quiz_questions'])}",
        )
        if st.button("Return to Menu", use_container_width=True):
            st.session_state["quiz_finished_final"] = False
            st.session_state["quiz_active"] = False
            for key in list(st.session_state.keys()):
                if key.startswith("scored_"):
                    del st.session_state[key]
            st.rerun()


# 6. Search Translation (Tekstowe - bez audio)
with search_translation:
    query_tr = st.text_input("Enter Translation Query", key="search_tr_in")
    search_triggered = st.button(
        "Search", use_container_width=True, key="search_tr_btn"
    )

    if search_triggered:
        df = list_translations(query_tr)
        if not df.empty:
            st.session_state["cached_tr_results"] = df.to_dict("records")
        else:
            st.session_state["cached_tr_results"] = []
            st.info("No translations found.")

    if (
        "cached_tr_results" in st.session_state
        and st.session_state["cached_tr_results"]
    ):
        st.header("Saved Translations")
        for idx, row in enumerate(st.session_state["cached_tr_results"]):
            with st.container(border=True):
                src_lang = row.get("Input_Language")
                tgt_lang = row.get("Target_Language")
                st.subheader(f"{src_lang} -> {tgt_lang}")
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("**Original:**")
                    st.info(row.get("Input_Text"))
                with c2:
                    st.markdown("**Translation:**")
                    st.success(row.get("Translation"))

# 7. Search Audio Translations (Tekstowe - bez audio)
with search_audio:
    query_audio = st.text_input(
        "Enter Audio Transcription Query", key="search_audio_in"
    )
    search_triggered_audio = st.button(
        "Search", use_container_width=True, key="search_audio_btn"
    )

    if search_triggered_audio:
        df_audio = list_audio_translations(query_audio)
        if not df_audio.empty:
            st.session_state["cached_audio_results"] = df_audio.to_dict("records")
        else:
            st.session_state["cached_audio_results"] = []
            st.info("No audio translations found.")

    if (
        "cached_audio_results" in st.session_state
        and st.session_state["cached_audio_results"]
    ):
        st.header("Saved Audio Translations")
        for idx, row in enumerate(st.session_state["cached_audio_results"]):
            with st.container(border=True):
                src_lang = row.get("Input_Language")
                tgt_lang = row.get("Target_Language")
                st.subheader(f"{src_lang} -> {tgt_lang}")

                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("**Transcription:**")
                    st.info(row.get("Transcription"))
                with c2:
                    st.markdown("**Translation:**")
                    st.success(row.get("Translation"))

# 8. Search Correction (Tekstowe - bez audio)
with search_correction:
    query_cor = st.text_input("Enter Correction Query", key="search_cor_in")
    search_triggered_cor = st.button(
        "Search", use_container_width=True, key="search_cor_btn"
    )

    if search_triggered_cor:
        df_cor = list_corrections(query_cor)
        if not df_cor.empty:
            st.session_state["cached_cor_results"] = df.to_dict("records")
        else:
            st.session_state["cached_cor_results"] = []
            st.info("No corrections found.")

    if (
        "cached_cor_results" in st.session_state
        and st.session_state["cached_cor_results"]
    ):
        st.header("Saved Corrections")
        for idx, row in enumerate(st.session_state["cached_cor_results"]):
            with st.container(border=True):
                st.subheader(row.get("Input_Language"))
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("**Original:**")
                    st.error(row.get("Input_Text"))
                with c2:
                    st.markdown("**Corrected:**")
                    st.success(row.get("Correction"))

                with st.expander("Explanations"):
                    st.write(row.get("Difficult_Words"))
                    st.divider()
                    st.write(row.get("Grammar_Rules"))
