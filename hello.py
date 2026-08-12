from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic

load_dotenv()

llm = ChatAnthropic(model="claude-sonnet-4-6")
print(llm.invoke("Say hello in one short sentence.").content)
