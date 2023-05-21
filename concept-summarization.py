import os

import tiktoken

from langchain.utilities import WikipediaAPIWrapper
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI, VertexAI
from langchain.chains import LLMChain

CONCEPT = "iPhone"

openai_llm = OpenAI(temperature=0.9, model_name="gpt-4")
google_llm = VertexAI()

def get_wikipedia_entry(concept):
    MAX_TOKENS = 4096
    wikipedia = WikipediaAPIWrapper()
    encoding = tiktoken.encoding_for_model("gpt-4")

    wikipedia_content = wikipedia.run(concept)
    entry = ""
    for wikipedia_line in wikipedia_content.split("\n"):
        if len(encoding.encode(entry)) + len(encoding.encode(wikipedia_line)) < MAX_TOKENS:
            entry += wikipedia_line + "\n"
    return entry

knowledge_check_prompt = PromptTemplate(
    input_variables=["concept"],
    template="Answer only \"yes\" or \"no\", can you confidently provide an initial release year for {concept}?",
)
from_scratch_prompt = PromptTemplate(
    input_variables=["concept"],
    template="In one sentence, {concept} is useful because it:",
)
from_wikipedia_prompt = PromptTemplate(
    input_variables=["concept", "wikipedia_entry"],
    template="Use the following Wikipedia entry to, in one sentence, summarize why {concept} is useful.\n\n{wikipedia_entry}",
)

openai_knowledge_check = LLMChain(llm=openai_llm, prompt=knowledge_check_prompt).run(CONCEPT).replace('\n', '')
google_knowledge_check = LLMChain(llm=google_llm, prompt=knowledge_check_prompt).run(CONCEPT).replace('\n', '')

if openai_knowledge_check.lower()[:2] == "no" and google_knowledge_check.lower()[:2] == "no":
    print("OpenAI:", openai_knowledge_check)
    print("Google:", google_knowledge_check)
    print("Using Wikipedia and OpenAI...")
    print(LLMChain(llm=openai_llm, prompt=from_wikipedia_prompt).run(concept=CONCEPT, wikipedia_entry=get_wikipedia_entry(CONCEPT)))

elif openai_knowledge_check.lower()[:2] == "no" and google_knowledge_check.lower()[:3] == "yes":
    print("OpenAI:", openai_knowledge_check)
    print("Google:", google_knowledge_check)
    print("Using Google...")
    print(LLMChain(llm=google_llm, prompt=from_scratch_prompt).run(CONCEPT))

elif openai_knowledge_check.lower()[:3] == "yes" and google_knowledge_check.lower()[:2] == "no":
    print("OpenAI:", openai_knowledge_check)
    print("Google:", google_knowledge_check)
    print("Using OpenAI...")
    print(LLMChain(llm=openai_llm, prompt=from_scratch_prompt).run(CONCEPT))

elif openai_knowledge_check.lower()[:3] == "yes" and google_knowledge_check.lower()[:3] == "yes":
    print("OpenAI:", openai_knowledge_check)
    print("Google:", google_knowledge_check)
    print("Using Google...")
    print(LLMChain(llm=google_llm, prompt=from_scratch_prompt).run(CONCEPT))

else:
    raise ValueError("Knowledge check prompt returned unexpected value.", openai_knowledge_check, google_knowledge_check)
