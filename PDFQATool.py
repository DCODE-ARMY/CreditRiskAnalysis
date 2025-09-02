import os
from typing import Type, List
from pydantic import BaseModel, Field
from urllib.parse import urlparse
from mistralai import Mistral
from crewai.tools import BaseTool
from dotenv import load_dotenv

load_dotenv()

class PDFQAToolInput(BaseModel):
    """Accepts 1–10 PDF paths or URLs plus a question."""
    pdf_paths: List[str] = Field(
        ..., 
        description="list of Local filesystem paths or public URLs to up to 10 scanned PDF files",
    )
    question: str = Field(
        ..., 
        description="The crew agent’s question to answer using the PDFs"
    )

class PDFQATool(BaseTool):
    name: str = "PDFQATool"
    description: str = (
        "Uploads scanned PDFs to Mistral, retrieves signed HTTPS URLs, "
        "and answers a question across all of them in one go."
    )
    args_schema: Type[PDFQAToolInput] = PDFQAToolInput

    def _run(self, pdf_paths, question) -> str:
        # 1. Initialize SDK client
        api_key = os.getenv("MISTRAL_API_KEY")
        if not api_key:
            raise ValueError("MISTRAL_API_KEY must be set in the environment")
        client = Mistral(api_key=api_key)

        # 2. For each local file, upload and get a signed HTTPS URL
        content_chunks = [{"type": "text", "text": question}]
        for path in pdf_paths:
            parsed = urlparse(path)
            if parsed.scheme in ("http", "https"):
                url_ref = path
            else:
                # Upload local PDF for OCR processing
                upload_resp = client.files.upload(
                    file={
                        "file_name": os.path.basename(path),
                        "content": open(path, "rb")
                    },
                    purpose="ocr"
                )  # :contentReference[oaicite:0]{index=0}
                # Get a signed HTTPS URL
                url_ref = client.files.get_signed_url(
                    file_id=upload_resp.id
                ).url  # :contentReference[oaicite:1]{index=1}

            content_chunks.append({
                "type": "document_url",
                "document_url": url_ref
            })

        # 3. Ask the model, which OCRs & understands all docs at once
        chat_resp = client.chat.complete(
            model="mistral-medium-latest",
            messages=[{"role": "user", "content": content_chunks}],
            temperature=0.0,
        )  # :contentReference[oaicite:2]{index=2}

        # 4. Return the aggregated answer
        return chat_resp.choices[0].message.content

# # Example usage
# if __name__ == "__main__":
#     tool = PDFQATool()
#     result = tool.run(
#         pdf_paths=["D:\\crew\\Version 3 - company house\\companies_house_docs\\01134945_3XUvnHT810InX5m_qlT2N_-ApZsmaKMAA0yrAid0Pt4.pdf",
#                    "companies_house_docs//01134945_IpMDdSWr1bJa0DmZRa1XA2N8IRSzG47KAykqJHZ8xPY.pdf",
#                    "D:\\crew\\Version 3 - company house\\companies_house_docs\\01134945_MJn2qDDwS_QUxTjwJFfVPhsmjQQZyOKPdHBqocqTZ4A.pdf"],
#         question="What kinds of information are in these documents? I want  you list them all. Please do not include any information about the company itself, just the types of information in the documents." 
#     )
#     print(result)
