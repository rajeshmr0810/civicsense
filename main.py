import os
import tiktoken
from openai import OpenAI, RateLimitError, APIError, AuthenticationError


class Chatbot:
    def __init__(self, api_key, model="gpt-4o-mini", temperature=0.7, max_tokens=100,
                 token_budget=1000, system_prompt="You are a helpful assistant."):
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.token_budget = token_budget
        self.messages = [{"role": "system", "content": system_prompt}]
        self.encoding = self._get_encoding()

    def _get_encoding(self):
        try:
            return tiktoken.encoding_for_model(self.model)
        except KeyError:
            return tiktoken.get_encoding("cl100k_base")

    def _count_tokens(self, text):
        return len(self.encoding.encode(text))

    def _total_tokens_used(self):
        try:
            return sum(self._count_tokens(msg["content"]) for msg in self.messages)
        except Exception as e:
            return 0

    def _enforce_token_budget(self):
        try:
            while self._total_tokens_used() > self.token_budget:
                if len(self.messages) <= 2:
                    break
                self.messages.pop(1)
        except Exception as e:
            print(f"[token budget error]: {e}")

    def chat(self, user_input):
        self.messages.append({"role": "user", "content": user_input})
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self.messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            reply = response.choices[0].message.content
            self.messages.append({"role": "assistant", "content": reply})
            self._enforce_token_budget()
            return reply
        except AuthenticationError:
            return "Error: Invalid API key."
        except RateLimitError:
            return "Error: Rate limit exceeded."
        except APIError as e:
            return f"API error: {e}"

    def get_token_count(self):
        return self._total_tokens_used()


# Usage
if __name__ == "__main__":
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Set OPENAI_API_KEY environment variable.")

    bot = Chatbot(api_key=api_key)
    print("Chatbot ready. Type 'exit' to quit.")

    while True:
        user_input = input("You: ")
        if user_input.strip().lower() in {"exit", "quit"}:
            break
        response = bot.chat(user_input)
        print(f"Assistant: {response}")
        print(f"Tokens used: {bot.get_token_count()}")   