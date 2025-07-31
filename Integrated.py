import streamlit as st
from transformers import pipeline, BartTokenizer
from transformers import PegasusTokenizer, PegasusForConditionalGeneration
import fitz  # PyMuPDF
import easyocr
import docx2txt
import tempfile
import os
import torch

# -----------------------------
# Load models
# -----------------------------
# BART
summarizer_bart = pipeline("summarization", model="facebook/bart-large-cnn", device=0)
bart_tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")

# Pegasus
pegasus_model_name = "google/pegasus-xsum"
pegasus_tokenizer = PegasusTokenizer.from_pretrained(pegasus_model_name)
pegasus_model = PegasusForConditionalGeneration.from_pretrained(pegasus_model_name).to(device)


# OCR reader
ocr_reader = easyocr.Reader(['en'])

# -----------------------------
# Helper functions
# -----------------------------
def extract_text_from_pdf(file_path):
    doc = fitz.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def extract_text_from_image(image_path):
    result = ocr_reader.readtext(image_path, detail=0)
    return " ".join(result)

def extract_text_from_docx(file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
        tmp.write(file.read())
        tmp_path = tmp.name
    return docx2txt.process(tmp_path)

def chunk_text_tokenwise(text, tokenizer, max_tokens=1024):
    words = text.split()
    chunks = []
    chunk = []

    current_len = 0
    for word in words:
        word_len = len(tokenizer.encode(word, add_special_tokens=False))
        if current_len + word_len > max_tokens:
            chunks.append(" ".join(chunk))
            chunk = [word]
            current_len = word_len
        else:
            chunk.append(word)
            current_len += word_len

    if chunk:
        chunks.append(" ".join(chunk))
    return chunks

def summarize_with_pegasus(text, max_length=120, min_length=30):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inputs = pegasus_tokenizer(text, return_tensors="pt", truncation=True, padding="longest").to(device)
    summary_ids = pegasus_model.generate(
        inputs["input_ids"],
        max_length=max_length,
        min_length=min_length,
        length_penalty=2.0,
        num_beams=4,
        early_stopping=True
    )
    return pegasus_tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Insurance Document Summarizer")
st.title("📄 Insurance Document Summarizer")

st.markdown("Upload a **PDF**, **image**, or **Word** file to extract and summarize its content.")

uploaded_file = st.file_uploader("📂 Choose a file", type=["pdf", "png", "jpg", "jpeg", "docx"])

model_choice = st.selectbox("🤖 Choose summarization model", [
    "BART (facebook/bart-large-cnn)",
    "Pegasus (google/pegasus-xsum)"
])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
        tmp_file.write(uploaded_file.read())
        temp_file_path = tmp_file.name

    if uploaded_file.name.endswith(".pdf"):
        extracted_text = extract_text_from_pdf(temp_file_path)
    elif uploaded_file.name.endswith(('.png', '.jpg', '.jpeg')):
        extracted_text = extract_text_from_image(temp_file_path)
    elif uploaded_file.name.endswith(".docx"):
        extracted_text = extract_text_from_docx(uploaded_file)
    else:
        extracted_text = ""

    if extracted_text:
        st.subheader("📑 Extracted Text")
        st.text_area("Raw Text", extracted_text, height=300)

        if st.button("🔍 Summarize"):
            with st.spinner("Summarizing..."):
                summaries = []

                if model_choice.startswith("BART"):
                    chunks = chunk_text_tokenwise(extracted_text, bart_tokenizer)
                    for chunk in chunks:
                        try:
                            if len(chunk.strip().split()) < 20:
                                continue
                            summary_chunk = summarizer_bart(
                                chunk, max_length=200, min_length=50, do_sample=False
                            )[0]['summary_text']
                            summaries.append(summary_chunk)
                        except Exception as e:
                            summaries.append(f"[Error: {str(e)}]")

                elif model_choice.startswith("Pegasus"):
                    chunks = chunk_text_tokenwise(extracted_text, pegasus_tokenizer, max_tokens=512)
                    for chunk in chunks:
                        try:
                            if len(chunk.strip().split()) < 20:
                                continue
                            summary_chunk = summarize_with_pegasus(chunk)
                            summaries.append(summary_chunk)
                        except Exception as e:
                            summaries.append(f"[Error: {str(e)}]")

                summary = "\n\n".join(summaries) if summaries else "No valid summary generated."

            st.subheader("🧠 Summary")
            st.success(summary)
    else:
        st.warning("❌ Could not extract any text from the uploaded file.")
