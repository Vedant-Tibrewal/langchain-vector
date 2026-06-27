import os

from dotenv import load_dotenv

load_dotenv()

if __name__ == "__main__":
    print("Hello ReAct LangGraph with Function Calling")
    print(os.getenv("LANGSMITH_PROJECT"))
