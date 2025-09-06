# main.py
# Simple Agents SDK demo using Gemini via OpenAI-compatible Chat Completions

import os

from agents import (
    Agent,
    Runner,
    AsyncOpenAI,
    set_default_openai_client,
    OpenAIChatCompletionsModel,
    set_tracing_disabled,
)

# 1) API key and model
api_key = "AIzaSyBbQqhTFCI5dME9zpqUBYGsKqracCynbs8"             # replace with your key
model = "gemini-2.5-flash-lite"           # your chosen Gemini model

# 2) Define an OpenAI-compatible client for Gemini
#    This base_url points to Google's OpenAI-compatible endpoint.
external_client = AsyncOpenAI(
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    api_key=api_key,
)
# Make it the default client so SDK calls use it.
set_default_openai_client(external_client)

# 3) Register the model (Chat Completions style) and disable tracing
#    Note: Agents SDK defaults to Responses API when using OpenAI models.
#    Since we are using an OpenAI-compatible Chat Completions backend (Gemini),
#    we explicitly set a Chat Completions model and turn off tracing (not supported here).
llm_model = OpenAIChatCompletionsModel(
    model=model,
    openai_client=external_client,
)

# Tracing is enabled by default for OpenAI; disable for Gemini
set_tracing_disabled(True)

# 4) Define the agent
agent = Agent(
    name="simple_agent",
    instructions="You are a helpful assistant, specialized in providing single-line responses.",
    model=llm_model,
)

# 5) Run the agent and print the answer
response = Runner.run_sync(agent, "What is AI?")
print(response.final_output)
