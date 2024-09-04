from transformers import MarianMTModel, MarianTokenizer
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
# Load the model and tokenizer
model_name_en = "Helsinki-NLP/opus-mt-ar-en"  # for English to Arabic
model_name_ar = "Helsinki-NLP/opus-mt-en-ar"  # for English to Arabic
en_tokenizer = MarianTokenizer.from_pretrained(model_name_en)
en_model = MarianMTModel.from_pretrained(model_name_en)
ar_tokenizer = MarianTokenizer.from_pretrained(model_name_ar)
ar_model = MarianMTModel.from_pretrained(model_name_ar)

def translateToEnglish(text):
    translated = en_model.generate(**en_tokenizer(text, return_tensors="pt", padding=True))
    translated_text = en_tokenizer.decode(translated[0], skip_special_tokens=True)
    return translated_text
def translateToArabic(text):
    translated = ar_model.generate(**ar_tokenizer(text, return_tensors="pt", padding=True))
    translated_text = ar_tokenizer.decode(translated[0], skip_special_tokens=True)
    return translated_text