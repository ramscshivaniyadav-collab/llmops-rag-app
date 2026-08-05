from dotenv import load_dotenv
from langfuse import get_client


# load api keys
load_dotenv()

#shifting the labels in dev
langfuse = get_client()

langfuse.update_prompt(
    name ='rag_app_system_prompt',
    version = None,
    new_labels=['staging']
)