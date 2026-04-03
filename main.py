from dotenv import load_dotenv
import os

load_dotenv()


def main():
    print("Hello from langchain-graph-aie!")
    has_openai_key = os.environ.get("OPENAI_API_KEY")
    if has_openai_key:
        print("OpenAI API key is set.")
    else:
        print("OpenAI API key is not set. Please set it in the .env file.")


if __name__ == "__main__":
    main()
