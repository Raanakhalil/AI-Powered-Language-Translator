import os
import gradio as gr
from groq import Groq

# ====== API Key Handling ======
api_key = os.environ.get("GROQ_API_KEY", "").strip()
if not api_key:
    raise ValueError("⚠️ GROQ_API_KEY is not set. Please add it in your Space settings.")

# ====== Config ======
GROQ_MODEL = "llama-3.3-70b-versatile"

# Complete global language list (major + regional)
LANGUAGES = [
    "Afrikaans", "Albanian", "Amharic", "Arabic", "Armenian", "Assamese", "Aymara", "Azerbaijani",
    "Bambara", "Basque", "Belarusian", "Bengali", "Bhojpuri", "Bosnian", "Bulgarian", "Burmese",
    "Catalan", "Cebuano", "Chinese (Simplified)", "Chinese (Traditional)", "Corsican", "Croatian",
    "Czech", "Danish", "Dhivehi", "Dogri", "Dutch", "English", "Esperanto", "Estonian", "Ewe",
    "Filipino", "Finnish", "French", "Frisian", "Galician", "Georgian", "German", "Greek",
    "Guarani", "Gujarati", "Haitian Creole", "Hausa", "Hawaiian", "Hebrew", "Hindi", "Hmong",
    "Hungarian", "Icelandic", "Igbo", "Ilocano", "Indonesian", "Irish", "Italian", "Japanese",
    "Javanese", "Kannada", "Kazakh", "Khmer", "Kinyarwanda", "Konkani", "Korean", "Krio", "Kurdish",
    "Kyrgyz", "Lao", "Latin", "Latvian", "Lingala", "Lithuanian", "Luganda", "Luxembourgish",
    "Macedonian", "Maithili", "Malagasy", "Malay", "Malayalam", "Maltese", "Maori", "Marathi",
    "Meiteilon (Manipuri)", "Mizo", "Mongolian", "Nepali", "Norwegian", "Nyanja", "Odia (Oriya)",
    "Oromo", "Pashto", "Persian", "Polish", "Portuguese", "Punjabi", "Quechua", "Romanian",
    "Russian", "Samoan", "Sanskrit", "Scots Gaelic", "Sepedi", "Serbian", "Sesotho", "Shona",
    "Sindhi", "Sinhala", "Slovak", "Slovenian", "Somali", "Spanish", "Sundanese", "Swahili",
    "Swedish", "Tajik", "Tamil", "Tatar", "Telugu", "Thai", "Tigrinya", "Tsonga", "Turkish",
    "Turkmen", "Twi", "Ukrainian", "Urdu", "Uyghur", "Uzbek", "Vietnamese", "Welsh", "Xhosa",
    "Yiddish", "Yoruba", "Zulu",    "Arabic", "Bengali", "Chinese (Simplified)", "Chinese (Traditional)", "Czech", "Danish",
    "Dutch", "English", "Finnish", "French", "German", "Greek", "Hebrew", "Hindi", "Hungarian",
    "Indonesian", "Italian", "Japanese", "Korean", "Malay", "Norwegian", "Polish", "Portuguese",
    "Romanian", "Russian", "Spanish", "Swedish", "Thai", "Turkish", "Ukrainian", "Urdu", "Vietnamese",
    "Sinhala (Sri Lanka)", "Tamil (Sri Lanka)"
]

SYSTEM_PROMPT = (
    "You are a professional translator. "
    "Translate the user's message into the requested target language. "
    "Preserve meaning, tone, names, numbers, and punctuation. "
    "If the text already appears to be in the target language, return it unchanged. "
    "Output ONLY the translation—no explanations."
)

# ====== Translation Function ======
def translate_text(text, target_language):
    if not text.strip():
        return "Please enter some text to translate."

    try:
        client = Groq(api_key=api_key)

        user_prompt = (
            f"Target language: {target_language}\n\n"
            f"Text:\n{text}"
        )

        completion = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
            max_tokens=1024,
        )

        return completion.choices[0].message.content.strip()

    except Exception as e:
        return f"❌ Error: {e}"

# ====== UI Design ======
with gr.Blocks(title=" AI Language Translator") as demo:
    gr.HTML(
        """
        <div style='text-align: center; padding: 20px; background-color: #1E293B; border-radius: 12px;'>
            <h1 style='color: #38BDF8; font-size: 2.5em;'>🌍 LingoAI 🌍 – AI-Powered Language Translator</h1>
            <p style='color: white; font-size: 1.2em;'>Effortlessly translate your text into <b>over 100 languages</b> using Groq’s lightning-fast Llama 3 AI models.</p>
        </div>
        """
    )

    with gr.Row():
        with gr.Column(scale=3):
            src_text = gr.Textbox(
                label="✏️ Source Text",
                placeholder="Type or paste the text you want to translate...",
                lines=8
            )
        with gr.Column(scale=1):
            target_lang = gr.Dropdown(
                LANGUAGES,
                value="Arabic",
                label="🌐 Target Language",
                interactive=True
            )
            translate_btn = gr.Button("🚀 Translate", variant="primary")

    out = gr.Textbox(label="📜 Translation", lines=8)

    translate_btn.click(
        translate_text,
        inputs=[src_text, target_lang],
        outputs=out
    )

    gr.HTML(
        """
        <div style='text-align: center; margin-top: 30px; color: gray;'>
            <p>⚡ Powered by <b>Groq</b> & Llama 3 AI – Built for speed, accuracy, and global communication.</p>
        </div>
        """
    )

if __name__ == "__main__":
    demo.launch()
