import requests
import json
import base64
from typing import Type
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
import os
from datetime import datetime
import time
from dotenv import load_dotenv
load_dotenv()
class CompaniesHouseToolInput(BaseModel):
    """Input schema for CompaniesHouseTool."""
    company_number: str = Field(default=None, description="Company number to search for (e.g., '01134945')")
    output_folder: str = Field(default="companies_house_docs", description="Folder to store downloaded account documents.")

class CompaniesHouseTool(BaseTool):
    name: str = "CompaniesHouseTool"
    description: str = """
        A CrewAI tool to fetch and download account filings, charges, and insolvency data from Companies House using a REST API key.
        Filters account filings to include only those from 2022 onward. 

        **Note**: This tool does not return the documents directly in the output. Instead, it downloads the relevant PDFs 
        (account filings from 2022 onward) and saves them to the specified output folder (default: 'companies_house_docs'). 
        The output JSON provides metadata about the downloaded files and other requested data (charges, insolvency).
    """
    args_schema: Type[BaseModel] = CompaniesHouseToolInput

    def __init__(self, ch_api_key: str = None, **kwargs):
        super().__init__(**kwargs)
        self._ch_api_key = os.getenv("COMPANYHOUSE_API_KEY")
        if not self._ch_api_key:
            raise ValueError("No Companies House REST API key provided. Set 'ch_api_key' or 'CH_API_KEY' environment variable.")
        
    def fetch_data(self, url: str) -> str:
        """Fetch data from Companies House API using Basic Auth."""
        auth_str = f"{self._ch_api_key}:"
        headers = {"Authorization": f"Basic {base64.b64encode(auth_str.encode()).decode()}"}
        response = requests.get(url, headers=headers)
        if response.status_code == 404:
            return "No data found (404)"
        response.raise_for_status()
        return response.text

    def fetch_document(self, document_id: str, output_path: str) -> None:
        """Download a document from Companies House using Basic Auth."""
        url = f"https://document-api.company-information.service.gov.uk/document/{document_id}/content"
        auth_str = f"{self._ch_api_key}:"
        headers = {"Authorization": f"Basic {base64.b64encode(auth_str.encode()).decode()}"}
        response = requests.get(url, headers=headers, stream=True)
        if response.status_code == 200:
            with open(output_path, "wb") as f:
                f.write(response.content)
                time.sleep(1)

    def _run(self, company_number: str = None, output_folder: str = "companies_house_docs") -> dict:
        if not company_number :
            return {"error": "Must provide  'company_number'."}


        os.makedirs(output_folder, exist_ok=True)
        try:
            public_data_base_url = "https://api.company-information.service.gov.uk"
            endpoints = {
                "accounts_filings": {
                    "url": f"{public_data_base_url}/company/{company_number}/filing-history",
                    "description": "Lists account-related filings (category: 'accounts') from 2022 onward, with PDFs downloaded to folder."
                },
                "charges": {
                    "url": f"{public_data_base_url}/company/{company_number}/charges",
                    "description": "Details all registered charges."
                },
                "insolvency": {
                    "url": f"{public_data_base_url}/company/{company_number}/insolvency",
                    "description": "Provides insolvency data."
                }
            }
            results = {}
            start_date = datetime(2022, 1, 1)
            for key, info in endpoints.items():
                try:
                    data = self.fetch_data(info["url"])
                    if key == "accounts_filings":
                        try:
                            parsed_data = json.loads(data)
                            account_items = [
                                item for item in parsed_data.get("items", [])
                                if item.get("category") == "accounts" and
                                datetime.strptime(item.get("date", "1900-01-01"), "%Y-%m-%d") >= start_date
                            ]
                            for item in account_items:
                                document_link = item.get("links", {}).get("document_metadata", "")
                                if document_link:
                                    document_id = document_link.split("/")[-1]
                                    output_path = os.path.join(output_folder, f"{company_number}_{document_id}.pdf")
                                    self.fetch_document(document_id, output_path)
                            data = json.dumps({"total_count": len(account_items), "items": account_items})
                        except json.JSONDecodeError:
                            data = "Error: Unable to parse filing history data"
                    results[key] = {
                        "description": info["description"],
                        "data": data
                    }
                except requests.exceptions.HTTPError as e:
                    results[key] = {
                        "description": info["description"],
                        "data": f"Error retrieving data: {e}"
                    }
                except Exception as e:
                    results[key] = {
                        "description": info["description"],
                        "data": f"Error retrieving data: {e}"
                    }
        except Exception as e:
            return {"error": f"Failed to process request: {e}"}
        return results

# # Example Usage
# CompaniesHouseToolInstance = CompaniesHouseTool()
# if __name__ == "__main__":
#     # Example usage
#     input_data = CompaniesHouseToolInput(company_number="01134945", output_folder="companies_house_docs")
#     result = CompaniesHouseToolInstance._run(company_number=input_data.company_number, output_folder=input_data.output_folder)
#     print(result)