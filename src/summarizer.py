import os
import openai
import semantic_kernel as sk
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion

kernel = sk.Kernel()

deployment, api_key, endpoint = sk.azure_openai_settings_from_dot_env()
kernel.add_text_completion_service("dv", OpenAIChatCompletion(deployment, endpoint, api_key))

prompt = kernel.create_semantic_function("""hello world""")

print(prompt())