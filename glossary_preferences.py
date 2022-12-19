import pdfplumber
from glossary_preferences_utils import select_glossaries

def isMachineReadable(pdf_file):
    try:
        pdf = pdfplumber.open(pdf_file)
    except:
        return

    for page_id in range(len(pdf.pages)):
        current_page = pdf.pages[page_id]
        words = current_page.extract_words()
        if(len(words)):
          break
    return len(words) > 0


# Returns preference order for english machine readable source, None otherwise
def get_preference_order(pdf_path, ocr_lang, src_lang, trans_lang, glossaries_path) :
    if src_lang != "en" or ocr_lang != "en" :
        return None

    if isMachineReadable(pdf_path) :
        return select_glossaries(pdf_path, src_lang, trans_lang, glossaries_path)
    return None
    
