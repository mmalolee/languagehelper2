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


#############
# CONSTANTS #
#############

MODEL = 'gpt-4o'
TEXT_TO_SPEECH_MODEL = 'tts-1'
SPEECH_TO_TEXT_MODEL = 'whisper-1'
EMBEDDING_MODEL = 'text-embedding-3-large'
EMBEDDING_DIM = 3072

QDRANT_COLLECTION_NAMES = ['translations', 'corrections']

LANGUAGES_INPUT = sorted(['English', 'German', 'Italian', 'Spanish', 'Czech', 'Polish'])
LANGUAGES_OUTPUT = sorted(['English', 'German', 'Italian', 'Spanish', 'Czech', 'Polish'])


#################
# OPENAI CLIENT #
#################

@st.cache_resource
def get_openai_client():
    return OpenAI(api_key=st.session_state['OPENAI_API_KEY'])


@st.cache_resource
def get_instructor_openai_client():
    return instructor.from_openai(get_openai_client())


##############
# BASEMODELS #
##############

class CheckGrammar(BaseModel):
    is_correct: bool


class Words(BaseModel):
    word: str
    w_explanation: str
    

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


class CheckLanguage(BaseModel):
    language_detected: str


#################################
# GETTING AUDIO BYTES FORM FILE #
#################################

def get_audio_bytes_from_file(file):
    audio_segment = AudioSegment.from_file(file)
    audio_segment_file_like = audio_segment.export(BytesIO())
    raw_bytes = audio_segment_file_like.getvalue()
    
    return raw_bytes


##########
# OPENAI #
##########

def get_audio_bytes_from_speech(speech):
    audio_segment_file_like = speech.export(BytesIO())
    raw_bytes = audio_segment_file_like.getvalue()
    
    return raw_bytes


def translate_text(text, language_input, language_output):
    # Używamy standardowego klienta OpenAI zamiast Langfuse
    openai_client = get_openai_client()

    translation = openai_client.chat.completions.create(
        model=MODEL,
        temperature=0,
        messages=
        [
            {
                'role': 'system',
                'content': f'''You are a professional translator that translates from {language_input} to {language_output}.
                Preserve formatting, punctuation, and capitalization exactly as in the input. Translate word by word if necessary. Output only the translated text.'''
            },
            {
                'role': 'user',
                'content': text.strip()
            }
        ]
    )

    return translation.choices[0].message.content


def check_language(to_check):
    # Używamy standardowego klienta Instructor zamiast Langfuse
    instructor_openai_client = get_instructor_openai_client()

    response = instructor_openai_client.chat.completions.create(
        model=MODEL,
        temperature=0,
        response_model=CheckLanguage,
        messages=
        [
            {
                'role': 'system',
                'content': f'''You are a language detector. Detect the language of the "{to_check}". Return detected language, e.g. "Polish". Do not return dialects, variations, or pidgin forms.
                Do not guess meanings of words in other languages.'''
            },
            {
                'role': 'user',
                'content': to_check
            }
        ])
    
    return response.language_detected


def check_grammar(to_check, input_language):
    instructor_openai_client = get_instructor_openai_client()

    response = instructor_openai_client.chat.completions.create(
        model=MODEL,
        temperature=0,
        response_model=CheckGrammar,
        messages=
        [
            {
                'role': 'system',
                'content': f'You are a professional grammar specialist in {input_language}. Evaluate the grammar of {to_check}.'
            },
            {
                'role': 'user',
                'content': to_check
            }
        ])
    return response.is_correct


def get_corr_words_and_grammar(to_correct, input_language):
    instructor_openai_client = get_instructor_openai_client()

    explanation = instructor_openai_client.chat.completions.create(
        model=MODEL,
        temperature=0,
        response_model=Explanation,
        messages=
        [
            {
                'role': 'system',
                'content': f'''You are a professional grammar specialist in {input_language}. Do 4 things:
                    1. Correct the grammar of the input text.
                    2. List all advanced or translation-challenging words from the "{to_correct}", and provide definitions or explanations in English.
                    3. Explain any difficult or tricky grammatical constructions found in the original text, such as unusual verb tenses, passive voice, subjunctive mood, or unusual word order.
                    4. All explanations, definitions, and translations must be given in English, regardless of the {input_language}.
                    '''
            },
            {
                'role': 'user',
                'content': to_correct.strip()
            }
        ]
    ).model_dump()

    corrected_form = explanation['corrected']
    difficult_words = explanation['difficult_words']
    grammar = explanation['grammars']

    return corrected_form, difficult_words, grammar


def text_to_speech(text):
    openai_client = get_openai_client()

    speech = openai_client.audio.speech.create(
        model=TEXT_TO_SPEECH_MODEL,
        voice='onyx',
        response_format='wav',
        input=text
    )

    return speech.content


def speech_to_text(speech):
    openai_client = get_openai_client() # Zmieniono na standardowego klienta, whisper nie wymaga instructora

    speech_file_like = BytesIO(speech)
    speech_file_like.name='audio.mp3'

    transcription = openai_client.audio.transcriptions.create(
        model=SPEECH_TO_TEXT_MODEL,
        file=speech_file_like,
        response_format='verbose_json',
    )

    return transcription.text


def translate_transcription(transcription, output_language):
    instructor_openai_client = get_instructor_openai_client()
    translation = instructor_openai_client.chat.completions.create(
        model=MODEL,
        temperature=0,
        response_model=AudioTranslationAndLanguage,
        messages=
        [
            {
                'role': 'system',
                'content': f'''You are a professional personal {output_language} translator. Do two things:
                1. Detect the language from "{transcription}" and translate to {output_language}. Preserve formatting, punctuation, and capitalization exactly as in the input. 
                Translate word by word if necessary. Output only the translated text.
                2. Return full detected from language "{transcription}." (e.g. Polish, German)'''
            },
            {
                'role': 'user',
                'content': transcription.strip()
            }
        ]
    )

    return translation.translation, translation.input_language


def get_embedding(text):
    openai_client = get_openai_client()

    result = openai_client.embeddings.create(
        input=[text],
        model=EMBEDDING_MODEL,
        dimensions=EMBEDDING_DIM
    )

    return result.data[0].embedding


##########
# QDRANT #
##########

@st.cache_resource
def get_qdrant_client():
    return QdrantClient(':memory:')


def assure_qdrant_collections_exist():
    qdrant_client = get_qdrant_client()

    if not qdrant_client.collection_exists(QDRANT_COLLECTION_NAMES[0]):
        print(f'Creating collection {QDRANT_COLLECTION_NAMES[0]}')
        qdrant_client.create_collection(
            collection_name=QDRANT_COLLECTION_NAMES[0],
            vectors_config={
                'Input_Language_Vector': VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
                'Input_Vector': VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
                'Translation_Language_Vector': VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
                'Translation_Vector': VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE)
            }
            )

    else:
        print(f'Collection {QDRANT_COLLECTION_NAMES[0]} exists')

    if not qdrant_client.collection_exists(QDRANT_COLLECTION_NAMES[1]):
        print(f'Creating collection {QDRANT_COLLECTION_NAMES[1]}')
        qdrant_client.create_collection(
            collection_name=QDRANT_COLLECTION_NAMES[1],
            vectors_config={
                'Input_Language_Vector': VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
                'Input_Text_Vector': VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
                'Correction_Vector': VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
                'Diff_Words_Vector': VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
                'Grammar_Rules_Vector': VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE)
            }
        )

    else:
        print(f'Collection {QDRANT_COLLECTION_NAMES[1]} exists')   


def add_translation_to_db(input_language, input, translation_language, translation):
    qdrant_client = get_qdrant_client()

    points_count = qdrant_client.count(
        collection_name=QDRANT_COLLECTION_NAMES[0],
        exact=True
    )

    qdrant_client.upsert(
        collection_name=QDRANT_COLLECTION_NAMES[0],
        points=[
            PointStruct(
                id=points_count.count+1,
                vector={
                    'Input_Language_Vector': get_embedding(input_language),
                    'Input_Vector': get_embedding(input),
                    'Translation_Language_Vector': get_embedding(translation_language),
                    'Translation_Vector': get_embedding(translation)
                },
                payload={
                    'Input_Text_Language': input_language,
                    'Input_Text': input,
                    'Translation_Language': translation_language,
                    'Translation': translation
                }
            )
        ]
    )


def add_correction_to_db(input_language, input, correction, diff_words, grammar_rules):
    qdrant_client = get_qdrant_client()

    points_count = qdrant_client.count(
        collection_name=QDRANT_COLLECTION_NAMES[1],
        exact=True
    )

    qdrant_client.upsert(
        collection_name=QDRANT_COLLECTION_NAMES[1],
        points=[
            PointStruct(
                id=points_count.count+1,
                vector={
                    'Input_Language_Vector': get_embedding(input_language),
                    'Input_Text_Vector': get_embedding(input),
                    'Correction_Vector': get_embedding(correction),
                    'Diff_Words_Vector': get_embedding(diff_words),
                    'Grammar_Rules_Vector': get_embedding(grammar_rules)
                },
                payload={
                    'Input_Language': input_language,
                    'Input_Text': input,
                    'Correction': correction,
                    'Diff_Words': diff_words,
                    'Grammar_Rules': grammar_rules
                }
            )
        ]
    )


def list_translations(query=None):
    qdrant_client = get_qdrant_client()

    if not query:
        translations_from_db = qdrant_client.scroll(collection_name=QDRANT_COLLECTION_NAMES[0], limit=10)[0]

        translations_results = []

        for translation_from_db in translations_from_db:
            translations_results.append({
                "Provided Input's Language": translation_from_db.payload['Input_Text_Language'],
                'Provided Input': translation_from_db.payload['Input_Text'],
                'Translation Language': translation_from_db.payload['Translation_Language'],
                'Translation': translation_from_db.payload['Translation'],
            })

        df_translations = pd.DataFrame(translations_results)

        return df_translations

    else:
        translations_from_db = qdrant_client.search(
            collection_name=QDRANT_COLLECTION_NAMES[0],
            query_vector= ('Input_Vector', get_embedding(query)),
            limit=10
        )

        translations_results_semantic = []

        for translation_from_db in translations_from_db:
            translations_results_semantic.append({
                'Similarity': round(translation_from_db.score, 2),
                'Input Text Language': translation_from_db.payload['Input_Text_Language'],
                'Input Text': translation_from_db.payload['Input_Text'],
                'Translation Language': translation_from_db.payload['Translation_Language'],
                'Translation': translation_from_db.payload['Translation']
            })

        df_translations_semantic = pd.DataFrame(translations_results_semantic)

        return df_translations_semantic
    

def list_corrections(query=None):
    qdrant_client = get_qdrant_client()

    if not query:
        corrections_from_db = qdrant_client.scroll(collection_name=QDRANT_COLLECTION_NAMES[1], limit=10)[0]

        correction_results = []

        for correction_from_db in corrections_from_db:
            correction_results.append({
                "Provided Input's Language": correction_from_db.payload['Input_Language'],
                'Provided Input': correction_from_db.payload['Input_Text'],
                'Correction': correction_from_db.payload['Correction'],
                'Difficult Words': correction_from_db.payload['Diff_Words'],
                'Grammar Rules': correction_from_db.payload['Grammar_Rules']
            })

        df_corrections = pd.DataFrame(correction_results)

        return df_corrections

    else:
        corrections_from_db = qdrant_client.search(
            collection_name=QDRANT_COLLECTION_NAMES[1],
            query_vector= ('Input_Text_Vector', get_embedding(query)),
            limit=10
        )

        corrections_results_semantic = []

        for correction_from_db in corrections_from_db:
            corrections_results_semantic.append({
                'Similarity': round(correction_from_db.score, 2),
                "Provided Input's Language": correction_from_db.payload['Input_Language'],
                'Provided Input': correction_from_db.payload['Input_Text'],
                'Correction': correction_from_db.payload['Correction'],
                'Difficult Words': correction_from_db.payload['Diff_Words'],
                'Grammar Rules': correction_from_db.payload['Grammar_Rules']
            })

        df_corrections_semantic = pd.DataFrame(corrections_results_semantic)

        return df_corrections_semantic


###################################
# TEXT TRANSLATION SESSION STATES #
###################################

if 'TT_input_language' not in st.session_state:
    st.session_state['TT_input_language'] = ''

if 'TT_output_language' not in st.session_state:
    st.session_state['TT_output_language'] = ''
    
if 'TT_detected_language' not in st.session_state:
    st.session_state['TT_detected_language'] = ''

if 'TT_translate_text_input' not in st.session_state:
    st.session_state['TT_translate_text_input'] = ''

if 'TT_translate_text_output' not in st.session_state:
    st.session_state['TT_translate_text_output'] = ''


####################################
# AUDIO TRANSLATION SESSION STATES #
####################################

if 'TA_output_language' not in st.session_state:
    st.session_state['TA_output_language'] = ''

if 'TA_input_language' not in st.session_state:
    st.session_state['TA_input_language'] = ''

if 'TA_input_transcription' not in st.session_state:
    st.session_state['TA_input_transcription'] = ''

if 'TA_output_text' not in st.session_state:
    st.session_state['TA_output_text'] = ''


##################################
# TEXT CORRECTION SESSION STATES #
##################################

if 'CT_input_language' not in st.session_state:
    st.session_state['CT_input_language'] = ''

if 'CT_input_text' not in st.session_state:
    st.session_state['CT_input_text'] = ''

if 'CT_output_text' not in st.session_state:
    st.session_state['CT_output_text'] = ''

if 'CT_diff_words' not in st.session_state:
    st.session_state['CT_diff_words'] = ''

if 'CT_grammar_rules' not in st.session_state:
    st.session_state['CT_grammar_rules'] = ''


####################################
# SPEECH CORRECTION SESSION STATES #
####################################

if 'CS_input_language' not in st.session_state:
    st.session_state['CS_input_language'] = ''

if 'CS_input_speech_as_text' not in st.session_state:
    st.session_state['CS_input_speech_as_text'] = ''

if 'CS_output_as_text' not in st.session_state:
    st.session_state['CS_output_as_text'] = ''

if 'CS_diff_words' not in st.session_state:
    st.session_state['CS_diff_words'] = ''

if 'CS_grammar_rules' not in st.session_state:
    st.session_state['CS_grammar_rules'] = ''


#########################
# API KEY SESSION STATE #
#########################

if 'OPENAI_API_KEY' not in st.session_state:
    st.session_state['OPENAI_API_KEY'] = ''


######################
# ASKING FOR API-KEY #
######################

if not st.session_state.get('OPENAI_API_KEY'):
    st.info('Enter your OpenAI API key')
    st.session_state['OPENAI_API_KEY'] = st.text_input('OpenAI API Key', type='password')

    if st.session_state['OPENAI_API_KEY']:
        st.rerun()
        
    st.stop()


########
# MAIN #
########

st.title('Language Helper')

assure_qdrant_collections_exist()

text_tr, audio_tr, text_cor, speech_cor, search_translation, search_correction = st.tabs(['Translate Text', 'Translate Audio', 'Correct Text', 'Record and Correct Speech', 'Search Translation', 'Search Correction'])


# Text translation #
with text_tr:
    c0, c1 = st.columns(2)
    translate_button = st.button('Translate Text', use_container_width=True)
    save_button = st.button('Save Translation', use_container_width=True)
     
    with c0:
        st.session_state['TT_input_language'] = st.selectbox('Select Input Language', LANGUAGES_INPUT, help='Choose the language you want translate from')
        st.session_state['TT_translate_text_input'] = st.text_area('Enter the Text', height=150, value=st.session_state['TT_translate_text_input']).strip()

    with c1:
        st.session_state['TT_output_language'] = st.selectbox('Select Output Language', LANGUAGES_OUTPUT, help='Choose the language you want translate to')

    if translate_button:

        if st.session_state['TT_translate_text_input']:
            st.session_state['TT_detected_language'] = check_language(st.session_state['TT_translate_text_input'])
            language_match = st.session_state['TT_detected_language'] == st.session_state['TT_input_language']

            if language_match:
                with c0:
                    input_text_audio = text_to_speech(st.session_state['TT_translate_text_input'])
                    st.audio(input_text_audio)

                with c1:
                    st.session_state['TT_translate_text_output'] = translate_text(st.session_state['TT_translate_text_input'], st.session_state['TT_input_language'], st.session_state['TT_output_language'])
                    translation_audio = text_to_speech(st.session_state['TT_translate_text_output'])
                    st.text_area('Translation', height=150, value=st.session_state['TT_translate_text_output'])
                    st.audio(translation_audio)

                st.toast('Text Has Been Translated!')

            else:
                with c1:
                    st.text_area('Translation', height=150, value='')

                st.warning(f"Input and Detected Language Don't Match. Switch Language Input to {st.session_state['TT_detected_language']}")

        else:
            with c1:
                st.text_area('Translation', height=150, value='')

            st.warning('Enter the Text')

    else:
        with c1:
            st.text_area('Translation', height=150, value='')

    if save_button:
        if st.session_state['TT_translate_text_output']:
            add_translation_to_db(
            st.session_state['TT_input_language'],
            st.session_state['TT_translate_text_input'],
            st.session_state['TT_output_language'],
            st.session_state['TT_translate_text_output']
        ) 
            
            st.toast('Translation Has Been Saved!')  
            st.session_state['TT_translate_text_output'] = ''

        else:
            st.warning('No Translation to Save')


# Audio translation #
with audio_tr:
    st.session_state['TA_output_language'] = st.selectbox('Select Desired Output Language', LANGUAGES_OUTPUT, help='Choose the language you want translate to')
    uploaded_file = st.file_uploader('Upload a file', type=['mp3', 'wav'])

    if uploaded_file:
        raw_audio_bytes = get_audio_bytes_from_file(uploaded_file)

        st.audio(raw_audio_bytes)

        translate_audio_button = st.button('Translate Audio Content', use_container_width=True)
        save_button = st.button('Save Audio Translation', use_container_width=True)

        if translate_audio_button:
            st.session_state['TA_input_transcription'] = speech_to_text(raw_audio_bytes)
            st.session_state['TA_output_text'], st.session_state['TA_input_language'] = translate_transcription(st.session_state['TA_input_transcription'], st.session_state['TA_output_language'])
            translated_audio = text_to_speech(st.session_state['TA_output_text'])

            with st.container(border=True):
                st.header(f"Provided audio is in {st.session_state['TA_input_language']}")
                st.text_area('Audio Translation', value=st.session_state['TA_output_text'])
                st.audio(translated_audio)

        if save_button:
            if st.session_state['TA_output_text']:
                add_translation_to_db(
                    st.session_state['TA_input_language'],
                    st.session_state['TA_input_transcription'],
                    st.session_state['TA_output_language'],
                    st.session_state['TA_output_text']
                )
                st.toast('Translation Has Been Saved!')  
                st.session_state['TA_output_text'] = ''

            else:
                st.warning('No Translation to Save')


# Text correction #
with text_cor:

    st.session_state['CT_input_language'] = st.selectbox('Select Language', LANGUAGES_INPUT, help='Select input language')
    st.session_state['CT_input_text'] = st.text_area('Enter the Sentence', height=75).strip()

    correct_button = st.button('Fix and Explain', use_container_width=True)
    save_button = st.button('Save Correction', use_container_width=True)

    if correct_button: 

        if st.session_state['CT_input_text']:
            detected_language = check_language(st.session_state['CT_input_text'])
            language_match = st.session_state['CT_input_language'] == detected_language

            if language_match:
                is_grammatically_correct = check_grammar(st.session_state['CT_input_text'], st.session_state['CT_input_language'])

                if is_grammatically_correct:
                    st.success('The Sentence is Correct')

                else:
                    st.session_state['CT_output_text'], difficult_words, grammars = get_corr_words_and_grammar(st.session_state['CT_input_text'], st.session_state['CT_input_language'])

                    sentence_audio = text_to_speech(st.session_state['CT_output_text'])

                    words_with_expl = ''
                    grammar_with_expl = ''

                    for word in difficult_words:
                        words_with_expl += f"""'{word['word'].capitalize()}' : {word['w_explanation'].capitalize()}\n"""
                    
                    if words_with_expl == '':
                        words_with_expl = 'No tricky words were detected.'

                    st.session_state['CT_diff_words'] = words_with_expl    

                    for grammar in grammars:
                        grammar_with_expl += f"""'{grammar["rule"]}' : {grammar["r_explanation"]}\n"""

                    if grammar_with_expl == '':
                        grammar_with_expl = 'No tricky words were detected.'

                    st.session_state['CT_grammar_rules'] = grammar_with_expl

                    with st.container(border=True):
                        st.header('Correct Form')
                        st.text_area('', value=st.session_state['CT_output_text'])
                    
                        with st.expander('Tricky Words Explanation'):
                            st.header('Tricky Words Explanation')
                            st.text_area('', value=st.session_state['CT_diff_words'])
                        
                        with st.expander('Grammar Explanation'):
                            st.header('Grammar Explanation')
                            st.text_area('', value=st.session_state['CT_grammar_rules'])

                        with st.expander('Pronunciation'):
                            st.header('Pronunciation')
                            st.audio(sentence_audio)
                            st.toast('Content Has Been Corrected!')
            else:
                st.warning(f"Input and Detected Language Don't Match. Switch Language Input to {detected_language}")

        else:
            st.warning('Enter the Content')

    if save_button:
        if st.session_state['CT_output_text']:
            add_correction_to_db(
                st.session_state['CT_input_language'],
                st.session_state['CT_input_text'],
                st.session_state['CT_output_text'],
                st.session_state['CT_diff_words'],
                st.session_state['CT_grammar_rules']
            )

            st.toast('Correction Has Been Saved!')  
            st.session_state['CT_output_text'] = ''

        else:
            st.warning('No Correction to Save')


# Speech correction #
with speech_cor:
    check_audio_correction_button = st.button('Check and Fix', use_container_width=True)
    save_button = st.button('Save Speech Correction', use_container_width=True)
    
    st.session_state['CS_input_language'] = st.selectbox('Select Recorded Speech Language', LANGUAGES_INPUT, help='Select input language')
    
    c0, c1 = st.columns([0.32, 0.68])

    with c0:
        input_speech = audiorecorder(start_prompt='Record Sentence', stop_prompt='Stop Recording Sentence')
    
    if not input_speech:

        if check_audio_correction_button:
            st.warning(f"Enter the Content")

    if input_speech:
        with c1:
            input_speech_bytes = get_audio_bytes_from_speech(input_speech)
            transcription = speech_to_text(input_speech_bytes)
            st.audio(input_speech_bytes)
        
        st.session_state['CS_input_speech_as_text'] = st.text_area('Check and Correct the Recording', value=transcription)

        language_detected = check_language(st.session_state['CS_input_speech_as_text'])
        language_match = language_detected == st.session_state['CS_input_language']

        if check_audio_correction_button:

            if language_match:
                is_grammatically_correct = check_grammar(st.session_state['CS_input_speech_as_text'], st.session_state['CS_input_language'])

                if is_grammatically_correct:
                    st.success('Sentence is Correct!')
                
                else:
                    sentence, difficult_words, grammars = get_corr_words_and_grammar(st.session_state['CS_input_speech_as_text'], st.session_state['CS_input_language'])

                    words_with_expl = ''
                    grammar_with_expl = ''

                    for word in difficult_words:
                        words_with_expl += f"""'{word['word'].capitalize()}' : {word['w_explanation'].capitalize()}\n"""

                    if words_with_expl == '':
                        words_with_expl = 'No tricky words were detected.'

                    for grammar in grammars:
                        grammar_with_expl += f"""'{grammar["rule"]}' : {grammar["r_explanation"]}\n"""

                    if grammar_with_expl == '':
                        grammar_with_expl = 'No tricky grammar rules were detected.'

                    st.session_state['CS_output_as_text'] = sentence
                    st.session_state['CS_diff_words'] = words_with_expl
                    st.session_state['CS_grammar_rules'] = grammar_with_expl

                    with st.container(border=True):
                        st.header('Correct Form')
                        st.text_area('', value=st.session_state['CS_output_as_text'])

                        with st.expander('Tricky Words'):
                            st.header('Tricky Words and Explanation')
                            st.text_area('', value=st.session_state['CS_diff_words'])

                        with st.expander('Grammar'):   
                            st.header('Grammar Explanation')
                            st.text_area('', value=st.session_state['CS_grammar_rules'])

                        with st.expander('Pronunciation'):
                            st.header('Pronunciation')
                            output_audio = text_to_speech(st.session_state['CS_output_as_text'])
                            st.audio(output_audio)

            else:
                st.warning(f"Input and Detected Language Don't Match. Switch Language Input to {language_detected}")


# Qdrant #
    if save_button:
        if st.session_state['CS_output_as_text']:
            add_correction_to_db(
                st.session_state['CS_input_language'],
                st.session_state['CS_input_speech_as_text'],
                st.session_state['CS_output_as_text'],
                st.session_state['CS_diff_words'],
                st.session_state['CS_grammar_rules']
            )

            st.toast('Correctionion Has Been Saved!')  
            st.session_state['CS_output_as_text'] = ''
            st.session_state['CS_input_speech_as_text'] = ''
        else:
            st.warning('No Correction to Save')


# Search translation #
with search_translation:
    query = st.text_input('Enter the Translation Query')
    search = st.button('Search Translation', use_container_width=True)
    
    if search:
        df_results = list_translations(query)
        
        if df_results.empty:
            st.info("No translations found locally.")
        else:
            st.header('Saved Translations')
            
            for index, row in df_results.iterrows():
                with st.container(border=True):
                    
                    col_top_1, col_top_2 = st.columns([0.8, 0.2])
                    with col_top_1:

                        try:
                            src = row["Provided Input's Language"]
                        except KeyError:
                            src = row["Input Text Language"]
                            
                        tgt = row['Translation Language']
                        st.subheader(f"{src} -> {tgt}")
                        
                    with col_top_2:
                        if 'Similarity' in row:
                            st.caption(f"Match: {row['Similarity']}")

                    c1, c2 = st.columns(2)
                    with c1:
                        st.markdown("**Original Text:**")
                        st.info(row['Provided Input'] if 'Provided Input' in row else row['Input Text'])
                    
                    with c2:
                        st.markdown("**Translation:**")
                        st.success(row['Translation'])


# Search correction #
with search_correction:
    query = st.text_input('Enter the Correction Query')
    search = st.button('Search Correction', use_container_width=True)
    
    if search:
        df_results = list_corrections(query)
        
        if df_results.empty:
            st.info("No corrections found locally.")
        else:
            st.header('Saved Corrections')
            
            for index, row in df_results.iterrows():
                with st.container(border=True):
                    
                    col_top_1, col_top_2 = st.columns([0.8, 0.2])
                    with col_top_1:
                        st.subheader(row['Provided Input\'s Language'])
                    with col_top_2:
                        if 'Similarity' in row:
                            st.caption(f"Match: {row['Similarity']}")

                    c1, c2 = st.columns(2)
                    with c1:
                        st.markdown("**Original Input:**")
                        st.error(row['Provided Input'])
                    
                    with c2:
                        st.markdown("**Corrected:**")
                        st.success(row['Correction'])

                    with st.expander("View Explanations"):
                        st.markdown("**Difficult Words**")
                        st.write(row['Difficult Words'])
                        
                        st.divider()
                        
                        st.markdown("**Grammar Rules**")
                        st.write(row['Grammar Rules'])