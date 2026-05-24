from indictrans import Transliterator

trn = Transliterator(source='en', target='or', build_lookup=True)

def translate_to_odia(text):
    return trn.transform(text)

def translate_to_english(text):
    # Use Google Translate API or IndicTrans
    return "translated_text"
