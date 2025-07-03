import os
import logging

from dotenv import load_dotenv
from httpx import ReadError
from supabase import create_client
from langchain_openai import OpenAIEmbeddings


load_dotenv()

supabase_client = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_SERVICE_KEY"))
embedder = OpenAIEmbeddings(model="text-embedding-3-small")  # embedding_dim = 1536

TABLES_NAMES_AND_FIELDS = {
    "engineer_equipment_list": ["name", "medium", "description" "utilities"],
    "lesson_learn": ["what", "why", "impact", "list_of_action"],
    "procurement_data": ["supplier_name", "item_name", "item_type", "category", "description"],
    "site_worker_presence": ["name", "status", "notes"]
}
ID_COLUMN = "id"


def populate_embeddings(overwrite: bool = False):
    """Populate embeddings for all tables in the Supabase database.

    If overwrite is True, embeddings will be recomputed and updated for all rows.
    If overwrite is False, only rows with null embeddings will be processed.
    """
    for table_name, text_fields in TABLES_NAMES_AND_FIELDS.items():
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
    overwrite = False

    if not overwrite:
        try:
            populate_embeddings(overwrite=overwrite)
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
            populate_embeddings(overwrite=overwrite)

    else:
        logging.info("Overwriting all embeddings in the database.")
        populate_embeddings(overwrite=True)