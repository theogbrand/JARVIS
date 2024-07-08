from prompts import reflection_prompt
from llm import LLM


def reflect_tool(user_thoughts, history):
    prompt = reflection_prompt.replace("USER_THOUGHTS", user_thoughts)
    ai_response = LLM(system_message=prompt).generate_response(messages=history)

    return ai_response