from typing import List

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

load_dotenv()

USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
INFO_URL = "https://en.wikipedia.org/wiki/Elon_Musk"

PROMPT_TEMPLATE = """
Given the information about a person, I want you to create:

1. A short summary
2. Two interesting facts about them

Here is the information about the person:

{information}
"""


class Output(BaseModel):
    summary: str
    facts: List[str]


def main():
    response = requests.get(INFO_URL, headers={"User-Agent": USER_AGENT})

    parser = BeautifulSoup(response.text, "html.parser")
    information = parser.get_text()

    summary_prompt_template = PromptTemplate(
        input_variables=["information"], template=PROMPT_TEMPLATE
    )

    llm = ChatOpenAI(temperature=0, model="gpt-5-nano").with_structured_output(Output)

    chain = summary_prompt_template | llm
    response = chain.invoke(input={"information": information})

    print(response)


if __name__ == "__main__":
    main()
