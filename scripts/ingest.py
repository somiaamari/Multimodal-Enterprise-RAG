import os
from app.ingestion.pipeline import ingest_document

if __name__ == "__main__":
    # Find the first PDF in the data folder
    data_dir = "data"
    pdf_files = [f for f in os.listdir(data_dir) if f.endswith('.pdf')]
    
    if not pdf_files:
        print("❌ No PDF files found in 'data/' folder.")
        print("   Please add a PDF file to the data/ directory.")
        exit(1)
    
    # Use the first PDF found
    pdf_path = os.path.join(data_dir, pdf_files[0])
    print(f"🔍 Found PDF: {pdf_files[0]}")
    
    # Ingest it
    index, count = ingest_document(pdf_path)
    
    print(f"\n🎉 Ingestion complete! {count} chunks indexed.")
    print(f"   You can now query your document.")