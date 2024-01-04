import os
import openai
import semantic_kernel as sk
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion

kernel = sk.Kernel()

openai.api_key = os.environ.get('OPENAI_API_KEY')
kernel.add_text_completion_service("dv", OpenAIChatCompletion("gpt-4",openai.api_key))

prompt = kernel.create_semantic_function("""1) A robot may not injure a human being or, through inaction, allow a human being to come to harm.
2) A robot must obey orders given it by human beings except where such orders would conflict with the First Law.
3) A robot must protect its own existence as long as such protection does not conflict with the First or Second Law.
                                         
Give me the TLDR in exactly 5 words.""")

#print(prompt())

summarize = kernel.create_semantic_function("{{$input}} \n\n one line TLDR with the fewest words")

print(summarize("""
1st Law of Thermodynamics - Energy cannot be created or destroyed.
2nd Law of Thermodynamics - For a spontaneous process, the entropy of the universe increases. 
3rd Law of Thermodynamics - A perfect crystal at zero Kelvin has zero entropy."""))
