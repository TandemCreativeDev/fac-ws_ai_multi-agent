from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END
from typing import TypedDict
from dotenv import load_dotenv
import datetime

load_dotenv()


class CodeReviewState(TypedDict):
    input: str
    code: str
    review: str
    refactored_code: str


llm = ChatOpenAI(model="gpt-4.1-nano")

coder_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a Senior Software Engineer. Write clean, well-structured Python code based on requirements."),
    ("human", "{input}")
])

reviewer_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a Code Reviewer. Provide constructive feedback focusing on readability, efficiency, and best practices."),
    ("human", "Review this code:\n{code}")
])

refactorer_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a Refactoring Expert. Implement the suggested improvements while maintaining functionality."),
    ("human",
     "Original code:\n{code}\n\nReview feedback:\n{review}\n\nRefactor accordingly:")
])


def coder_agent(state: CodeReviewState) -> CodeReviewState:
    response = llm.invoke(coder_prompt.format_messages(input=state["input"]))
    return {"code": response.content}


def reviewer_agent(state: CodeReviewState) -> CodeReviewState:
    response = llm.invoke(reviewer_prompt.format_messages(code=state["code"]))
    return {"review": response.content}


def refactorer_agent(state: CodeReviewState) -> CodeReviewState:
    response = llm.invoke(refactorer_prompt.format_messages(
        code=state["code"], review=state["review"]))
    return {"refactored_code": response.content}


def export_to_markdown(result: dict, task: str, pattern_name: str):

    filename = f"{pattern_name.upper()}.md"

    markdown_content = f"""# {pattern_name.replace('_', ' ').title()} Results

**Generated:** {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}  
**Task:** {task}

## Original Code

{result.get('code', 'No code generated')}

## Review Feedback

{result.get('review', 'No review available')}

## Refactored Code

{result.get('refactored_code', 'No refactored code available')}

---
*Generated using LangGraph Sequential Workflow Pattern*
"""

    with open(filename, 'w', encoding='utf-8') as f:
        f.write(markdown_content)

    print(f"✅ Results exported to: {filename}")


builder = StateGraph(CodeReviewState)
builder.add_node("coder", coder_agent)
builder.add_node("reviewer", reviewer_agent)
builder.add_node("refactorer", refactorer_agent)

builder.add_edge(START, "coder")
builder.add_edge("coder", "reviewer")
builder.add_edge("reviewer", "refactorer")
builder.add_edge("refactorer", END)

workflow = builder.compile()

if __name__ == "__main__":
    task = "Write a function that validates email addresses using regex"

    print("Running sequential workflow...")
    result = workflow.invoke({"input": task})

    export_to_markdown(result, task, "sequential_workflow")

    print("=== WORKFLOW COMPLETED ===")
    print("Check the generated markdown file for detailed results.")
