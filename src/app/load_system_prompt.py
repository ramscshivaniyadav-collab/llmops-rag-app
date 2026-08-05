# just use to understand the load functionality . The below code  is covered in rag_workflow
from dotenv import load_dotenv
from langfuse import get_client

# load the api keys
load_dotenv()

langfuse = get_client()

system_prompt = langfuse.get_prompt(
    name = 'rag_app_system_prompt',
    type='text',
    label='latest'
    )

print(system_prompt.version)
print(system_prompt.prompt)
print(system_prompt.config)
print(system_prompt.labels)



