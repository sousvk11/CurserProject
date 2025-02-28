import pandas as pd
import docx
import PyPDF2
import os
from typing import List, Dict

class DocumentProcessor:
    def __init__(self, training_doc_path: str):
        self.training_doc_path = training_doc_path
    
    def read_txt(self, file_path: str) -> str:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    def read_pdf(self, file_path: str) -> str:
        text = ""
        with open(file_path, 'rb') as f:
            pdf_reader = PyPDF2.PdfReader(f)
            for page in pdf_reader.pages:
                text += page.extract_text()
        return text
    
    def read_docx(self, file_path: str) -> str:
        doc = docx.Document(file_path)
        return '\n'.join([paragraph.text for paragraph in doc.paragraphs])
    
    def read_excel_csv(self, file_path: str) -> str:
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        else:
            df = pd.read_excel(file_path)
        return df.to_string()
    
    def process_all_documents(self) -> List[str]:
        documents = []
        for file in os.listdir(self.training_doc_path):
            file_path = os.path.join(self.training_doc_path, file)
            try:
                if file.endswith('.txt'):
                    documents.append(self.read_txt(file_path))
                elif file.endswith('.pdf'):
                    documents.append(self.read_pdf(file_path))
                elif file.endswith('.docx'):
                    documents.append(self.read_docx(file_path))
                elif file.endswith(('.xlsx', '.xls', '.csv')):
                    documents.append(self.read_excel_csv(file_path))
            except Exception as e:
                print(f"Error processing {file}: {str(e)}")
        return documents 