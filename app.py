import streamlit as st
from openai import OpenAI
import instructor
from pydantic import BaseModel, Field
from io import BytesIO
from pydub import AudioSegment
from audiorecorder import audiorecorder
from qdrant_client import QdrantClient
from qdrant_client import models
import pandas as pd
import os
import random

# ---------------------------------------------------------
# KONFIGURACJA STRONY
# ---------------------------------------------------------
st.set_page_config(layout="wide", page_title="Language Helper")

st.markdown(
    """
    <style>
        .block-container {
            max-width: 95%;
            padding-top: 1rem;
            padding-right: 1rem;
            padding-left: 1rem;
            padding-bottom: 1rem;
        }
        
        button[data-baseweb="tab"] {
            font-size: 14px;
            white-space: nowrap;
        }
        
        .similarity-badge {
            background-color: #f0f2f6;
            color: #31333F;
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 0.8em;
            font-weight: bold;
            margin-left: 10px;
            border: 1px solid #d6d6d6;
        }
        
        /* Centrowanie nagłówków w fiszkach */
        .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
            text-align: center !important;
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


class VerifiedSynonyms(BaseModel):
    synonyms: list[str] = Field(
        ...,
        description="List of words from the candidates that are actual synonyms or close alternatives.",
    )


class ClozeQuestion(BaseModel):
    sentence_with_blank: str = Field(
        ...,
        description="The sentence with one key grammatical word replaced by '______'.",
    )
    missing_word: str = Field(..., description="The correct word that was removed.")
    options: list[str] = Field(
        ...,
        description="3 incorrect options (distractors) that are grammatically plausible but wrong.",
    )
    context_clue: str = Field(
        ...,
        description="Grammatical clues resolving ambiguity WITHOUT translating the whole sentence (e.g. 'Subject: We (plural) | Tense: Past').",
    )


#################################
# AUDIO PROCESSING FUNCTIONS    #
#################################


def get_audio_bytes_from_file(file):
    audio_segment = AudioSegment.from_file(file)
    audio_segment_file_like = audio_segment.export(BytesIO(), format="mp3")
    return audio_segment_file_like.getvalue()


def get_audio_bytes_from_speech(speech):
    audio_segment_file_like = speech.export(BytesIO(), format="mp3")
    return audio_segment_file_like.getvalue()


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
                "content": f"""You are a language detector. Detect the language of the "{to_check}". Return detected language name in English (e.g. 'Polish', 'German'). Do not return dialects, variations, or pidgin forms.""",
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
                       - 'word': Provide the **base dictionary form (lemma)** of the word in {input_language}.
                       - 'translation': The direct, short translation in English.
                       - 'definition': A short explanation or definition in English to provide context.
                    3. Explain any difficult or tricky grammatical constructions found in the original text.
                    4. All explanations must be given in English.
                    """,
            },
            {"role": "user", "content": to_correct.strip()},
        ],
    ).model_dump()

    return (
        explanation["corrected"],
        explanation["difficult_words"],
        explanation["grammars"],
    )


@st.cache_data(show_spinner=False)
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
                2. Return full detected language name in English (e.g. Polish, German).
                3. Extract difficult vocabulary, phrasal verbs, or idioms from the SOURCE text.
                   - For 'word': provide the base dictionary form (lemma).
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
            if name == "translations":
                vectors = {
                    "Input_Vector": models.VectorParams(
                        size=EMBEDDING_DIM, distance=models.Distance.COSINE
                    ),
                    "Input_Language_Vector": models.VectorParams(
                        size=EMBEDDING_DIM, distance=models.Distance.COSINE
                    ),
                    "Translation_Language_Vector": models.VectorParams(
                        size=EMBEDDING_DIM, distance=models.Distance.COSINE
                    ),
                    "Translation_Vector": models.VectorParams(
                        size=EMBEDDING_DIM, distance=models.Distance.COSINE
                    ),
                }
            elif name == "corrections":
                vectors = {
                    "Input_Text_Vector": models.VectorParams(
                        size=EMBEDDING_DIM, distance=models.Distance.COSINE
                    ),
                    "Input_Language_Vector": models.VectorParams(
                        size=EMBEDDING_DIM, distance=models.Distance.COSINE
                    ),
                    "Correction_Vector": models.VectorParams(
                        size=EMBEDDING_DIM, distance=models.Distance.COSINE
                    ),
                    "Diff_Words_Vector": models.VectorParams(
                        size=EMBEDDING_DIM, distance=models.Distance.COSINE
                    ),
                    "Grammar_Rules_Vector": models.VectorParams(
                        size=EMBEDDING_DIM, distance=models.Distance.COSINE
                    ),
                }
            else:
                vectors = {
                    "Transcription_Vector": models.VectorParams(
                        size=EMBEDDING_DIM, distance=models.Distance.COSINE
                    ),
                    "Input_Language_Vector": models.VectorParams(
                        size=EMBEDDING_DIM, distance=models.Distance.COSINE
                    ),
                    "Translation_Language_Vector": models.VectorParams(
                        size=EMBEDDING_DIM, distance=models.Distance.COSINE
                    ),
                    "Translation_Vector": models.VectorParams(
                        size=EMBEDDING_DIM, distance=models.Distance.COSINE
                    ),
                }
            qdrant_client.create_collection(
                collection_name=name, vectors_config=vectors
            )


def add_translation_to_db(input_language, input, translation_language, translation):
    qdrant_client = get_qdrant_client()
    points_count = qdrant_client.count(collection_name="translations", exact=True).count
    qdrant_client.upsert(
        collection_name="translations",
        points=[
            models.PointStruct(
                id=points_count + 1,
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
    if not words_list:
        return
    qdrant_client = get_qdrant_client()
    for item in words_list:
        points_count = qdrant_client.count(
            collection_name="translations", exact=True
        ).count
        word = item.word if hasattr(item, "word") else item.get("word", "")
        trans = (
            item.translation
            if hasattr(item, "translation")
            else item.get("translation", "")
        )
        if word and trans:
            qdrant_client.upsert(
                collection_name="translations",
                points=[
                    models.PointStruct(
                        id=points_count + 1,
                        vector={
                            "Input_Language_Vector": get_embedding(input_language),
                            "Input_Vector": get_embedding(word),
                            "Translation_Language_Vector": get_embedding(
                                translation_language
                            ),
                            "Translation_Vector": get_embedding(trans),
                        },
                        payload={
                            "Input_Text_Language": input_language,
                            "Input_Text": word,
                            "Translation_Language": translation_language,
                            "Translation": trans,
                        },
                    )
                ],
            )


def add_correction_to_db(input_language, input, correction, diff_words, grammar_rules):
    qdrant_client = get_qdrant_client()
    points_count = qdrant_client.count(collection_name="corrections", exact=True).count
    qdrant_client.upsert(
        collection_name="corrections",
        points=[
            models.PointStruct(
                id=points_count + 1,
                vector={
                    "Input_Language_Vector": get_embedding(input_language),
                    "Input_Text_Vector": get_embedding(input),
                    "Correction_Vector": get_embedding(correction),
                    "Diff_Words_Vector": get_embedding(str(diff_words)),
                    "Grammar_Rules_Vector": get_embedding(str(grammar_rules)),
                },
                payload={
                    "Input_Language": input_language,
                    "Input_Text": input,
                    "Correction": correction,
                    "Diff_Words": str(diff_words),
                    "Grammar_Rules": str(grammar_rules),
                },
            )
        ],
    )


def add_audio_translation_to_db(
    input_language, transcription, translation_language, translation
):
    qdrant_client = get_qdrant_client()
    points_count = qdrant_client.count(
        collection_name="audio_translations", exact=True
    ).count
    qdrant_client.upsert(
        collection_name="audio_translations",
        points=[
            models.PointStruct(
                id=points_count + 1,
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


# --- FUNKCJE WYSZUKIWANIA ---


def list_translations(query=None, language_filter=None, target_language_filter=None):
    client = get_qdrant_client()
    conditions = []

    if language_filter and language_filter != "All":
        conditions.append(
            models.FieldCondition(
                key="Input_Text_Language",
                match=models.MatchValue(value=language_filter),
            )
        )
    if target_language_filter and target_language_filter != "All":
        conditions.append(
            models.FieldCondition(
                key="Translation_Language",
                match=models.MatchValue(value=target_language_filter),
            )
        )

    search_filter = models.Filter(must=conditions) if conditions else None

    if not query:
        results, _ = client.scroll(
            collection_name="translations", scroll_filter=search_filter, limit=10
        )
        return pd.DataFrame(
            [
                {
                    "Input_Language": r.payload["Input_Text_Language"],
                    "Input_Text": r.payload["Input_Text"],
                    "Target_Language": r.payload["Translation_Language"],
                    "Translation": r.payload["Translation"],
                }
                for r in results
            ]
        )
    else:
        search_result = client.search(
            collection_name="translations",
            query_vector=("Input_Vector", get_embedding(query)),
            query_filter=search_filter,
            limit=10,
        )
        return pd.DataFrame(
            [
                {
                    "Score": round(r.score, 2),
                    "Input_Language": r.payload["Input_Text_Language"],
                    "Input_Text": r.payload["Input_Text"],
                    "Target_Language": r.payload["Translation_Language"],
                    "Translation": r.payload["Translation"],
                }
                for r in search_result
            ]
        )


def list_corrections(query=None):
    client = get_qdrant_client()
    if not query:
        results, _ = client.scroll(collection_name="corrections", limit=10)
        # --- POPRAWIONY BŁĄD KEYERROR (Diff_Words) ---
        return pd.DataFrame(
            [
                {
                    "Input_Language": r.payload["Input_Language"],
                    "Input_Text": r.payload["Input_Text"],
                    "Correction": r.payload["Correction"],
                    "Difficult_Words": r.payload.get("Diff_Words", "N/A"),
                    "Grammar_Rules": r.payload["Grammar_Rules"],
                }
                for r in results
            ]
        )
    else:
        search_result = client.search(
            collection_name="corrections",
            query_vector=("Input_Text_Vector", get_embedding(query)),
            limit=10,
        )
        # --- POPRAWIONY BŁĄD KEYERROR (Diff_Words) ---
        return pd.DataFrame(
            [
                {
                    "Score": round(r.score, 2),
                    "Input_Language": r.payload["Input_Language"],
                    "Input_Text": r.payload["Input_Text"],
                    "Correction": r.payload["Correction"],
                    "Difficult_Words": r.payload.get("Diff_Words", "N/A"),
                    "Grammar_Rules": r.payload["Grammar_Rules"],
                }
                for r in search_result
            ]
        )


def list_audio_translations(query=None):
    client = get_qdrant_client()
    if not query:
        results, _ = client.scroll(collection_name="audio_translations", limit=10)
        return pd.DataFrame(
            [
                {
                    "Input_Language": r.payload["Input_Language"],
                    "Transcription": r.payload["Transcription"],
                    "Translation": r.payload["Translation"],
                }
                for r in results
            ]
        )
    else:
        search_result = client.search(
            collection_name="audio_translations",
            query_vector=("Transcription_Vector", get_embedding(query)),
            limit=10,
        )
        return pd.DataFrame(
            [
                {
                    "Score": round(r.score, 2),
                    "Input_Language": r.payload["Input_Language"],
                    "Transcription": r.payload["Transcription"],
                    "Translation": r.payload["Translation"],
                }
                for r in search_result
            ]
        )


def get_quiz_data(lang1, lang2):
    client = get_qdrant_client()
    points, _ = client.scroll(collection_name="translations", limit=1000)
    candidates = []

    target_langs = {lang1, lang2}

    for p in points:
        p_in = p.payload.get("Input_Text_Language")
        p_out = p.payload.get("Translation_Language")
        text_in = p.payload.get("Input_Text", "")
        text_out = p.payload.get("Translation", "")

        if {p_in, p_out} == target_langs or (
            p_in in target_langs and p_out in target_langs
        ):
            if len(text_in.split()) <= 4:
                candidates.append(
                    {
                        "question": text_in,
                        "answer": text_out,
                        "direction": f"{p_in} -> {p_out}",
                        "target_lang": p_out,
                    }
                )
                candidates.append(
                    {
                        "question": text_out,
                        "answer": text_in,
                        "direction": f"{p_out} -> {p_in}",
                        "target_lang": p_in,
                    }
                )
    return candidates


# --- FUNKCJA GENERUJĄCA PYTANIA DO QUIZU GRAMATYCZNEGO (CLOZE) ---
def generate_grammar_quiz_questions(language, count=5):
    client = get_qdrant_client()
    points, _ = client.scroll(
        collection_name="corrections",
        scroll_filter=models.Filter(
            must=[
                models.FieldCondition(
                    key="Input_Language", match=models.MatchValue(value=language)
                )
            ]
        ),
        limit=50,
    )

    if len(points) < 3:
        return []

    selected_points = random.sample(points, min(len(points), count))
    quiz_questions = []

    instructor_client = get_instructor_openai_client()

    for p in selected_points:
        correct_sentence = p.payload["Correction"]

        try:
            cloze = instructor_client.chat.completions.create(
                model=MODEL,
                temperature=0.7,
                response_model=ClozeQuestion,
                messages=[
                    {
                        "role": "system",
                        "content": f"""You are a strict language teacher creating a grammar quiz for {language}. 
                        Task:
                        1. Take the provided sentence.
                        2. Replace ONE key grammatical word with '______'.
                        3. CRITICAL: Analyze the sentence structure. Identify the Subject, Tense, Gender, or Number required to fill the gap correctly.
                        4. Provide these details as 'context_clue' (e.g. 'Subject: We (plural) | Tense: Past'). DO NOT TRANSLATE the whole sentence.
                        5. Generate 3 distractors that are plausible but wrong for THIS specific grammatical context.""",
                    },
                    {"role": "user", "content": f"Sentence: {correct_sentence}"},
                ],
            )

            options = cloze.options + [cloze.missing_word]
            random.shuffle(options)

            quiz_questions.append(
                {
                    "question": cloze.sentence_with_blank,
                    "answer": cloze.missing_word,
                    "options": options,
                    "context_clue": cloze.context_clue,
                    "explanation": p.payload.get(
                        "Grammar_Rules", "No explanation available."
                    ),
                    "original_wrong": p.payload.get("Input_Text", ""),
                }
            )
        except Exception as e:
            print(f"Error generating cloze: {e}")
            continue

    return quiz_questions


# --- FUNKCJA WERYFIKACJI ALTERNATYW ---
def get_semantic_alternatives(target_text, target_lang):
    client = get_qdrant_client()

    search_result = client.search(
        collection_name="translations",
        query_vector=("Input_Vector", get_embedding(target_text)),
        score_threshold=0.70,
        limit=15,
    )

    raw_candidates = set()
    for r in search_result:
        payload = r.payload
        if payload.get("Input_Text_Language") == target_lang:
            word = payload.get("Input_Text", "").strip()
            if word.lower() != target_text.lower().strip():
                raw_candidates.add(word)
        if payload.get("Translation_Language") == target_lang:
            word = payload.get("Translation", "").strip()
            if word.lower() != target_text.lower().strip():
                raw_candidates.add(word)

    candidates_list = list(raw_candidates)
    if not candidates_list:
        return []

    try:
        instructor_client = get_instructor_openai_client()
        response = instructor_client.chat.completions.create(
            model=MODEL,
            temperature=0,
            response_model=VerifiedSynonyms,
            messages=[
                {
                    "role": "system",
                    "content": f"You are a lexicographer for {target_lang}. Given a target word and a list of candidates, return ONLY those candidates that are valid synonyms or very close semantic alternatives. Ignore typos or unrelated words. If none match, return an empty list.",
                },
                {
                    "role": "user",
                    "content": f"Target Word: {target_text}\nLanguage: {target_lang}\nCandidates: {candidates_list}",
                },
            ],
        )
        return response.synonyms
    except Exception:
        return []


##################
# SESSION STATES #
##################

keys = [
    "TT_input_language",
    "TT_output_language",
    "TT_translate_text_input",
    "TT_translate_text_output",
    "TT_audio_in",
    "TT_audio_out",
    "TA_output_language",
    "TA_input_language",
    "TA_input_transcription",
    "TA_output_text",
    "TA_audio_out",
    "TA_raw_diff_words",
    "CT_input_language",
    "CT_input_text",
    "CT_output_text",
    "CT_diff_words",
    "CT_grammar_rules",
    "CT_audio_out",
    "CT_raw_diff_words",
    "CS_input_language",
    "CS_input_speech_as_text",
    "CS_output_as_text",
    "CS_diff_words",
    "CS_grammar_rules",
    "CS_audio_out",
    "CS_raw_diff_words",
    "quiz_active",
    "quiz_questions",
    "quiz_current_index",
    "quiz_score",
    "quiz_submitted",
    "quiz_finished_final",
    "quiz_mode",
    "grammar_quiz_active",
    "grammar_quiz_questions",
    "grammar_quiz_index",
    "grammar_quiz_score",
    "grammar_quiz_submitted",
    "grammar_quiz_finished",
    "cached_tr_results",
    "cached_cor_results",
    "cached_audio_results",
    "search_tr_mode",
    "flashcard_idx",
    "flashcards_data",
    "flashcard_revealed",
    "flashcard_score",
    "quiz_alternatives",
]

for k in keys:
    if k not in st.session_state:
        if "audio" in k:
            st.session_state[k] = None
        elif (
            "list" in k
            or "words" in k
            or "results" in k
            or "questions" in k
            or "data" in k
            or "alternatives" in k
        ):
            st.session_state[k] = []
        elif "index" in k or "score" in k or "idx" in k:
            st.session_state[k] = 0
        elif "active" in k or "submitted" in k or "finished" in k or "revealed" in k:
            st.session_state[k] = False
        else:
            st.session_state[k] = ""

# API Key
if "OPENAI_API_KEY" not in st.session_state:
    st.session_state["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "")

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
    grammar_quiz_tab,
    search_tr,
    search_cor,
    search_aud,
) = st.tabs(
    [
        "Translate Text",
        "Translate Audio",
        "Correct Text",
        "Record & Correct",
        "Vocabulary Quiz",
        "Grammar Quiz",
        "Search Translations",
        "Search Corrections",
        "Staudio",
    ]
)

# ... (TABS 1-4 SAME AS BEFORE) ...
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

        if st.session_state.get("TA_output_text"):
            with st.container(border=True):
                st.header(
                    f"Provided audio is in {st.session_state['TA_input_language']}"
                )
                st.text_area(
                    "Audio Translation", value=st.session_state["TA_output_text"]
                )
                if st.session_state.get("TA_raw_diff_words"):
                    with st.expander("Extracted Vocabulary (will be added to Quiz)"):
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

# 5. Vocabulary Quiz
with quiz_tab:
    st.header("Vocabulary & Flashcards")

    if not st.session_state.get("quiz_active", False) and not st.session_state.get(
        "quiz_finished_final", False
    ):
        st.subheader("Settings")
        c1, c2 = st.columns(2)
        with c1:
            q_lang1 = st.selectbox("Language 1", LANGUAGES_INPUT, key="q_lang1")
        with c2:
            q_lang2 = st.selectbox("Language 2", LANGUAGES_OUTPUT, key="q_lang2")

        candidates = get_quiz_data(q_lang1, q_lang2)
        count = len(candidates)

        c_mode, c_stat = st.columns([0.6, 0.4])
        with c_mode:
            quiz_mode = st.radio(
                "Select Mode",
                ["Written Answer", "Select Translation", "Flashcards"],
                horizontal=True,
            )
        with c_stat:
            st.write("")
            color = "#4CAF50" if count >= 3 else "#FFC107"
            bg = "rgba(76, 175, 80, 0.1)" if count >= 3 else "rgba(255, 193, 7, 0.1)"
            text = (
                f"Found {count} items."
                if count >= 3
                else f"Not enough items ({count}). Need 3+."
            )

            st.markdown(
                f"""
                <div style="
                    background-color: {bg}; 
                    color: {color}; 
                    padding: 8px 15px; 
                    border-radius: 5px; 
                    border: 1px solid {color}; 
                    text-align: center; 
                    margin-top: 27px; /* Wyrównanie do kropek radio */
                    font-weight: bold;
                ">
                    {text}
                </div>
                """,
                unsafe_allow_html=True,
            )

        if q_lang1 == q_lang2:
            st.error("Please select two different languages.")
        elif count >= 3:
            num_q = st.slider(
                "Select number of items", 1, min(count, 20), min(count, 5)
            )

            if st.button("Start", use_container_width=True):
                selected = random.sample(candidates, num_q)

                if quiz_mode == "Select Translation":
                    for item in selected:
                        same_dir_candidates = [
                            c for c in candidates if c["direction"] == item["direction"]
                        ]
                        other = [
                            c["answer"]
                            for c in same_dir_candidates
                            if c["answer"] != item["answer"]
                        ]
                        pool_size = len(other)
                        distractors = random.sample(other, min(3, pool_size))
                        options = distractors + [item["answer"]]
                        random.shuffle(options)
                        item["options"] = options

                st.session_state["quiz_questions"] = selected
                st.session_state["quiz_mode"] = quiz_mode
                st.session_state["quiz_current_index"] = 0
                st.session_state["quiz_score"] = 0
                st.session_state["quiz_active"] = True
                st.session_state["quiz_submitted"] = False

                if quiz_mode == "Flashcards":
                    st.session_state["flashcard_revealed"] = False
                    st.session_state["flashcard_score"] = 0

                st.rerun()

    # --- ACTIVE SESSION ---
    elif st.session_state.get("quiz_active", False):
        idx = st.session_state["quiz_current_index"]
        questions = st.session_state["quiz_questions"]
        current = questions[idx]
        mode = st.session_state["quiz_mode"]

        st.write(f"Item {idx + 1} of {len(questions)}")
        st.progress((idx) / len(questions))

        # --- LOGIKA FISZEK (FLASHCARDS) ---
        if mode == "Flashcards":
            with st.container(border=True):
                st.caption(f"Translate to: {current['direction'].split(' -> ')[1]}")
                st.markdown(f"## {current['question']}")

                if st.session_state["flashcard_revealed"]:
                    st.divider()
                    st.markdown(
                        f"<h2 style='color: #4CAF50;'>{current['answer']}</h2>",
                        unsafe_allow_html=True,
                    )

                    st.audio(text_to_speech(current["answer"]), format="audio/mpeg")

                    with st.spinner("Checking for synonyms..."):
                        alts = get_semantic_alternatives(
                            current["answer"], current["target_lang"]
                        )
                        if alts:
                            st.info(f"💡 Also correct: {', '.join(alts)}")

            if not st.session_state["flashcard_revealed"]:
                if st.button("Reveal Answer", use_container_width=True):
                    st.session_state["flashcard_revealed"] = True
                    st.rerun()
            else:
                c_know, c_dontknow = st.columns(2)
                with c_know:
                    if st.button("✅ I knew it", use_container_width=True):
                        st.session_state["flashcard_score"] += 1
                        if idx < len(questions) - 1:
                            st.session_state["quiz_current_index"] += 1
                            st.session_state["flashcard_revealed"] = False
                            st.rerun()
                        else:
                            st.session_state["quiz_score"] = st.session_state[
                                "flashcard_score"
                            ]
                            st.session_state["quiz_active"] = False
                            st.session_state["quiz_finished_final"] = True
                            st.rerun()
                with c_dontknow:
                    if st.button("❌ Didn't know", use_container_width=True):
                        if idx < len(questions) - 1:
                            st.session_state["quiz_current_index"] += 1
                            st.session_state["flashcard_revealed"] = False
                            st.rerun()
                        else:
                            st.session_state["quiz_score"] = st.session_state[
                                "flashcard_score"
                            ]
                            st.session_state["quiz_active"] = False
                            st.session_state["quiz_finished_final"] = True
                            st.rerun()

            st.divider()
            if st.button("Exit Quiz", type="secondary", use_container_width=True):
                st.session_state["quiz_active"] = False
                st.rerun()

        # --- LOGIKA QUIZU (WRITTEN / SELECT) ---
        else:
            with st.container(border=True):
                st.caption(f"Translate to: {current['direction'].split(' -> ')[1]}")
                st.subheader(f"'{current['question']}'")

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
                    is_correct = False
                    if user_ans.strip().lower() == current["answer"].strip().lower():
                        is_correct = True
                    elif mode == "Written Answer":
                        with st.spinner("Checking synonyms..."):
                            synonyms = get_semantic_alternatives(
                                current["answer"], current["target_lang"]
                            )
                            synonyms_lower = [s.lower() for s in synonyms]
                            if user_ans.strip().lower() in synonyms_lower:
                                is_correct = True
                                st.success(
                                    f"Correct! The primary answer is '{current['answer']}', but you used a valid synonym."
                                )

                    if is_correct:
                        if f"scored_{idx}" not in st.session_state:
                            st.session_state["quiz_score"] += 1
                            st.session_state[f"scored_{idx}"] = True
                        if (
                            user_ans.strip().lower()
                            == current["answer"].strip().lower()
                        ):
                            st.success(f"Correct. Answer: {current['answer']}")
                    else:
                        st.error(f"Incorrect. Answer: {current['answer']}")
                        with st.spinner("Checking DB for alternatives..."):
                            alts = get_semantic_alternatives(
                                current["answer"], current["target_lang"]
                            )
                            if alts:
                                st.info(
                                    f"💡 Valid alternatives found in DB: {', '.join(alts)}"
                                )

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

            st.divider()
            if st.button("Exit Quiz", type="secondary", use_container_width=True):
                st.session_state["quiz_active"] = False
                st.rerun()

    # --- RESULT SCREEN ---
    elif st.session_state.get("quiz_finished_final", False):
        st.header("Results")
        st.balloons()
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

# 6. Grammar Quiz (NEW TAB)
with grammar_quiz_tab:
    st.header("Grammar Correction Quiz (Fill in the blank)")

    if not st.session_state.get(
        "grammar_quiz_active", False
    ) and not st.session_state.get("grammar_quiz_finished", False):
        st.subheader("Settings")
        g_lang = st.selectbox("Select Language for Correction Quiz", LANGUAGES_INPUT)

        # Sprawdź dostępność pytań
        client = get_qdrant_client()
        points_count = client.count(
            collection_name="corrections",
            count_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="Input_Language", match=models.MatchValue(value=g_lang)
                    )
                ]
            ),
            exact=True,
        ).count

        c_info, c_btn = st.columns([0.6, 0.4])
        with c_btn:
            st.write("")
            color = "#4CAF50" if points_count >= 3 else "#FFC107"
            bg = (
                "rgba(76, 175, 80, 0.1)"
                if points_count >= 3
                else "rgba(255, 193, 7, 0.1)"
            )
            text = (
                f"Found {points_count} items."
                if points_count >= 3
                else f"Not enough ({points_count}). Need 3+."
            )

            st.markdown(
                f"""
                <div style="
                    background-color: {bg}; 
                    color: {color}; 
                    padding: 8px 15px; 
                    border-radius: 5px; 
                    border: 1px solid {color}; 
                    text-align: center; 
                    margin-top: 27px;
                    font-weight: bold;
                ">
                    {text}
                </div>
                """,
                unsafe_allow_html=True,
            )

        if points_count >= 3:
            num_q = st.slider(
                "Select number of questions",
                1,
                min(points_count, 10),
                min(points_count, 5),
            )
            if st.button("Start Grammar Quiz", use_container_width=True):
                # --- KLUCZOWA POPRAWKA: RESETOWANIE STANU ---
                st.session_state["grammar_quiz_questions"] = []

                with st.spinner("Generating grammar questions using AI..."):
                    questions = generate_grammar_quiz_questions(g_lang, num_q)
                    if questions:
                        st.session_state["grammar_quiz_questions"] = questions
                        st.session_state["grammar_quiz_index"] = 0
                        st.session_state["grammar_quiz_score"] = 0
                        st.session_state["grammar_quiz_active"] = True
                        st.session_state["grammar_quiz_submitted"] = False
                        st.rerun()
                    else:
                        st.error("Failed to generate questions. Try again.")

    # --- ACTIVE GRAMMAR QUIZ ---
    elif st.session_state.get("grammar_quiz_active", False):
        idx = st.session_state["grammar_quiz_index"]
        questions = st.session_state["grammar_quiz_questions"]
        current = questions[idx]

        st.write(f"Question {idx + 1} of {len(questions)}")
        st.progress((idx) / len(questions))

        with st.container(border=True):
            # Wyświetlamy kontekst z zabezpieczeniem .get()
            context_clue = current.get("context_clue", "No clue available")
            st.info(f"ℹ️ **Grammatical Clue:** {context_clue}")

            st.markdown(f"### {current['question']}")

            user_ans = st.radio(
                "Choose the missing word:",
                current["options"],
                key=f"g_radio_{idx}",
                disabled=st.session_state["grammar_quiz_submitted"],
            )

            if not st.session_state["grammar_quiz_submitted"]:
                if st.button("Check", use_container_width=True):
                    st.session_state["grammar_quiz_submitted"] = True
                    st.rerun()
            else:
                if user_ans == current["answer"]:
                    st.success(f"Correct! The word was **{current['answer']}**.")
                    if f"g_scored_{idx}" not in st.session_state:
                        st.session_state["grammar_quiz_score"] += 1
                        st.session_state[f"g_scored_{idx}"] = True
                else:
                    st.error(
                        f"Incorrect. The correct word was **{current['answer']}**."
                    )

                with st.expander("Why? (Context & Explanation)"):
                    st.write(f"**Original Mistake:** {current['original_wrong']}")
                    st.info(current["explanation"])

                st.audio(
                    text_to_speech(
                        current["question"].replace("______", current["answer"])
                    ),
                    format="audio/mpeg",
                )

                if idx + 1 < len(questions):
                    if st.button("Next Question", use_container_width=True):
                        st.session_state["grammar_quiz_index"] += 1
                        st.session_state["grammar_quiz_submitted"] = False
                        st.rerun()
                else:
                    if st.button("Finish Quiz", use_container_width=True):
                        st.session_state["grammar_quiz_active"] = False
                        st.session_state["grammar_quiz_finished"] = True
                        st.rerun()

        st.divider()
        if st.button("Exit Grammar Quiz", type="secondary", use_container_width=True):
            st.session_state["grammar_quiz_active"] = False
            st.rerun()

    # --- GRAMMAR RESULT ---
    elif st.session_state.get("grammar_quiz_finished", False):
        st.header("Grammar Quiz Results")
        st.balloons()
        st.metric(
            "Final Score",
            f"{st.session_state['grammar_quiz_score']} / {len(st.session_state['grammar_quiz_questions'])}",
        )
        if st.button("Return to Menu", use_container_width=True):
            st.session_state["grammar_quiz_finished"] = False
            st.session_state["grammar_quiz_active"] = False
            for key in list(st.session_state.keys()):
                if key.startswith("g_scored_"):
                    del st.session_state[key]
            st.rerun()

# 7. Search Translation (Dynamic Layout + Score)
with search_tr:
    cur_mode = st.session_state.get("search_tr_mode", "None")

    if cur_mode == "None":
        cols = st.columns([0.7, 0.3])
    elif cur_mode == "Both":
        cols = st.columns([0.4, 0.2, 0.2, 0.2])
    else:
        cols = st.columns([0.5, 0.25, 0.25])

    with cols[0]:
        query_tr = st.text_input(
            "Translation Query",
            key="search_tr_in",
            label_visibility="collapsed",
            placeholder="Enter text to search...",
        )
    with cols[1]:
        filter_mode = st.selectbox(
            "Filter By",
            ["None", "Input Language", "Target Language", "Both"],
            key="search_tr_mode",
            label_visibility="collapsed",
        )

    f_in = f_out = None
    if cur_mode == "Input Language":
        with cols[2]:
            f_in = st.selectbox(
                "In Lang", LANGUAGES_INPUT, key="dyn_in", label_visibility="collapsed"
            )
    elif cur_mode == "Target Language":
        with cols[2]:
            f_out = st.selectbox(
                "Out Lang",
                LANGUAGES_OUTPUT,
                key="dyn_out",
                label_visibility="collapsed",
            )
    elif cur_mode == "Both":
        with cols[2]:
            f_in = st.selectbox(
                "In Lang",
                LANGUAGES_INPUT,
                key="dyn_in_both",
                label_visibility="collapsed",
            )
        with cols[3]:
            f_out = st.selectbox(
                "Out Lang",
                LANGUAGES_OUTPUT,
                key="dyn_out_both",
                label_visibility="collapsed",
            )

    if st.button("Search", use_container_width=True, key="search_tr_btn"):
        df = list_translations(query_tr, f_in, f_out)
        st.session_state["cached_tr_results"] = df.to_dict("records")
        if df.empty:
            st.warning("No results found.")

    if st.session_state.get("cached_tr_results"):
        st.header("Saved Translations")
        for idx, row in enumerate(st.session_state["cached_tr_results"]):
            with st.container(border=True):
                # Badge z wynikiem
                badge = (
                    f"<span class='similarity-badge'>Score: {row.get('Score')}</span>"
                    if "Score" in row
                    else ""
                )
                st.markdown(
                    f"**{row['Input_Language']} &rarr; {row['Target_Language']}** {badge}",
                    unsafe_allow_html=True,
                )
                c1, c2 = st.columns(2)
                with c1:
                    st.caption("Original")
                    st.info(row.get("Input_Text"))
                with c2:
                    st.caption("Translation")
                    st.success(row.get("Translation"))

# 8. Search Corrections
with search_cor:
    query_cor = st.text_input("Enter Correction Query", key="search_cor_in")
    if st.button("Search", use_container_width=True, key="search_cor_btn"):
        df = list_corrections(query_cor)
        st.session_state["cached_cor_results"] = df.to_dict("records")
        if df.empty:
            st.warning("No results found.")

    if st.session_state.get("cached_cor_results"):
        st.header("Saved Corrections")
        for idx, row in enumerate(st.session_state["cached_cor_results"]):
            with st.container(border=True):
                badge = (
                    f"<span class='similarity-badge'>Score: {row.get('Score')}</span>"
                    if "Score" in row
                    else ""
                )
                st.markdown(
                    f"**{row['Input_Language']}** {badge}", unsafe_allow_html=True
                )
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

# 9. Search Audio
with search_aud:
    query_audio = st.text_input(
        "Enter Audio Transcription Query", key="search_audio_in"
    )
    if st.button("Search", use_container_width=True, key="search_audio_btn"):
        df = list_audio_translations(query_audio)
        st.session_state["cached_audio_results"] = df.to_dict("records")
        if df.empty:
            st.warning("No results found.")

    if st.session_state.get("cached_audio_results"):
        st.header("Saved Audio Translations")
        for idx, row in enumerate(st.session_state["cached_audio_results"]):
            with st.container(border=True):
                badge = (
                    f"<span class='similarity-badge'>Score: {row.get('Score')}</span>"
                    if "Score" in row
                    else ""
                )
                st.markdown(
                    f"**{row['Input_Language']} (Audio)** {badge}",
                    unsafe_allow_html=True,
                )
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("**Transcription:**")
                    st.info(row.get("Transcription"))
                with c2:
                    st.markdown("**Translation:**")
                    st.success(row.get("Translation"))
