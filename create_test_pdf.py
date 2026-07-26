from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import cm

def create_test_pdf(filename):
    c = canvas.Canvas(filename, pagesize=A4)
    width, height = A4

    c.setFont("Helvetica-Bold", 24)
    c.drawString(3*cm, height - 3*cm, "RAG System Test Document")

    c.setFont("Helvetica", 12)
    y_position = height - 5*cm

    content = [
        "Introduction to RAG",
        "",
        "RAG stands for Retrieval-Augmented Generation. It is a powerful",
        "technique that combines the capabilities of large language models",
        "with external knowledge retrieval. This allows the model to generate",
        "more accurate and contextually relevant responses.",
        "",
        "How RAG Works",
        "",
        "1. Document Loading: The system loads documents from various sources",
        "   such as PDFs, websites, or databases.",
        "",
        "2. Text Chunking: Large documents are split into smaller chunks",
        "   that can be efficiently processed and embedded.",
        "",
        "3. Embedding Generation: Each text chunk is converted into a",
        "   vector representation using machine learning models.",
        "",
        "4. Vector Storage: These embeddings are stored in a vector database",
        "   for fast similarity search.",
        "",
        "5. Retrieval: When a user asks a question, the system finds the",
        "   most relevant document chunks.",
        "",
        "6. Generation: The retrieved context is combined with the user's",
        "   question and sent to the LLM for answer generation.",
        "",
        "Benefits of RAG",
        "",
        "- Up-to-date information: RAG can access recent documents",
        "- Reduced hallucinations: Answers are grounded in actual documents",
        "- Transparency: Users can see which documents were used",
        "- Cost-effective: Doesn't require model retraining",
        "",
        "This test document is used to verify the functionality of the",
        "PDF RAG system you have just set up.",
    ]

    for line in content:
        if y_position < 3*cm:
            c.showPage()
            y_position = height - 3*cm
            c.setFont("Helvetica", 12)

        if line.startswith("Introduction") or line.startswith("How RAG") or \
           line.startswith("Benefits"):
            c.setFont("Helvetica-Bold", 14)
            c.drawString(3*cm, y_position, line)
            c.setFont("Helvetica", 12)
        else:
            c.drawString(3*cm, y_position, line)

        y_position -= 0.6*cm

    c.save()
    print(f"PDF created: {filename}")

if __name__ == "__main__":
    create_test_pdf("d:/代码/PDF-ptokect/data/pdfs/test_document.pdf")
