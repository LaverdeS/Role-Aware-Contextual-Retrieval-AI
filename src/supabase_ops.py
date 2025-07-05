import os
import logging

from dotenv import load_dotenv
from httpx import ReadError
from supabase import create_client
from langchain_openai import AzureOpenAIEmbeddings


load_dotenv()

supabase_client = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_SERVICE_KEY"))
embedder = AzureOpenAIEmbeddings(
    azure_deployment=os.getenv("AZURE_EMBEDDING_DEPLOYMENT_NAME"),
    openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    model=os.getenv("AZURE_EMBEDDING_MODEL_NAME", "text-embedding-3-small")  # Default to text-embedding-3-small if not specified
)

SUPABASE_TABLES = {
    "engineer_equipment_list": {
        "text_fields": ["name", "medium", "description", "utilities"],
        "output_fields": ["id", "name", "medium", "description", "utilities", "status", "quantity"]
    },
    "lesson_learn": {
        "text_fields": ["what", "why", "impact", "list_of_action"],
        "output_fields": ["id", "what", "why", "impact", "list_of_action"]
    },
    "procurement_data": {
        "text_fields": ["supplier_name", "item_name", "item_type", "category", "description"],
        "output_fields": ["id", "supplier_name", "item_name", "item_type", "category", "description", "quantity", "unit", "unit_price", "total_cost", "status"]
    },
    "site_worker_presence": {
        "text_fields": ["name", "status", "notes"],
        "output_fields": ["id", "name", "status", "notes"]
    }
}

ID_COLUMN = "id"
TEXT_FIELDS_SEPARATOR = " || "


def populate_embeddings(overwrite: bool = False):
    """Populate embeddings for all tables in the Supabase database.

    If overwrite is True, embeddings will be recomputed and updated for all rows.
    If overwrite is False, only rows with null embeddings will be processed.
    """
    for table_name, metadata in SUPABASE_TABLES.items():
        text_fields = metadata["text_fields"]
        logging.info(f"Populating embeddings for table: {table_name}")
        response = supabase_client.table(table_name).select("*").execute()

        if overwrite:
            rows = response.data  # Process all rows
            logging.info(f"Overwriting embeddings for all {len(rows)} rows.\n")
        else:
            rows = [r for r in response.data if r.get("embedding") is None]
            logging.info(f"Found {len(rows)} rows without embeddings.\n")

        for row in rows:
            row_id = row[ID_COLUMN]
            text = " ".join(str(row.get(f, "")) for f in text_fields).strip()
            logging.info(f"Processing row ID: {row_id} with text: {text}")

            if not text:
                continue

            row_embedding = embedder.embed_query(text)
            supabase_client.table(table_name).update({"embedding": row_embedding}).eq(ID_COLUMN, row_id).execute()
            logging.info(f"✅ Updated row {row_id}\n")



if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    overwrite_embeddings = False

    if not overwrite_embeddings:
        try:
            populate_embeddings()
        except ReadError:
            logging.warning(
                "Unstable or throttled network (e.g., public Wi-Fi)."
                "\nOR Faulty VPN, proxy, or antivirus tampering with requests."
                "\nOR Requests to external services (e.g., OpenAIEmbeddings) being rate-limited or dropped mid-stream"
                "\nRetrying in 5 seconds..."
            )
            import time
            time.sleep(5)
            supabase_client = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_SERVICE_KEY"))
            populate_embeddings()

    else:
        logging.info("Overwriting all embeddings in the database.")
        populate_embeddings(overwrite=overwrite_embeddings)