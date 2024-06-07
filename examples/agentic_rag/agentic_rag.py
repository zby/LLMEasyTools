from llm_easy_tools import get_tool_defs, process_response
from openai import OpenAI
from pprint import pprint

from examples.agentic_rag.markdown_search import MarkdownSearchEngine


MAX_ITERATIONS = 5

SYS_PROMPT = """
Please answer the user query using the available tools and the documents.
To search for a document use a word or two that are likely to be found in a document on a subject related to the user query.
After you have found the document, you can use the lookup tool to find text inside the document.
After you know the answer, use the finish tool to answer the user query.
"""

class AgenticRAG:
    def __init__(self, query):
        markdown_search = MarkdownSearchEngine('indexdir', 'docs')
        self.markdown_search = markdown_search
        self.client = OpenAI()
        self.answer = None
        self.step = 0
        self.query = query
        self.messages = [
            {"role": "system", "content": SYS_PROMPT},
            {"role": "user", "content": f"The user query is: {self.query}"}
        ]
        self.tools = [markdown_search.search, markdown_search.lookup, self.finish]
        
    def finish(self, answer: str):
        self.answer = answer
        
    def one_step(self):
        print(f"Step {self.step}")
        tool_schemas = get_tool_defs(self.tools)
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo-1106",
            messages=self.messages,
            tools=tool_schemas,
            tool_choice="auto",
        )
        message = response.choices[0].message
        print(f"LLM Message:\n\n{message}\n\n")
        self.messages.append(message)
        # There might be more than one tool calls in a single response so results are a list
        results = process_response(response, self.tools)
        if len(results) == 0:
            self.finish(message.content)
            return
        result = results[0]
        if result.error:
            raise Exception(f"Error: {result.error}")
        if not result.name == "finish":
            self.messages.append(result.to_message())
            print(f"Tool Message:\n\n{result.to_message()}\n\n")
        
    def run(self):
        while self.step < MAX_ITERATIONS and self.answer is None:
            self.one_step()
            self.step += 1
        return self.answer

if __name__ == "__main__":
    #agentic_rag = AgenticRAG("What is the atomic weight of oxygen?")
    agentic_rag = AgenticRAG("What is the full name of the Milhouse character in the Simpsons?")
    #agentic_rag = AgenticRAG("What is the maximum height of Heigh Plains in the US?")
    agentic_rag.run()
    print(f"Answer:\n{agentic_rag.answer}")

