import openai
import prompt_templates


def get_client(base_url, api_key):
    """
    Return an OpenAI client object.
    """
    return openai.OpenAI(base_url=base_url, api_key=api_key)


def compile_prompt(user_template, system_template, template_variables, **kwargs):
    """
    Compile an openAI prompt template.
    `user_template` should be a Python string with variable placeholders in {var} format.
    Pass var_name=var_value as kwargs.
    """
    if not system_template:
        system_template = "You are a helpful coding assistant. Return only Lean code, do not return any reasoning."
    messages_template = [
        {"role": "system", "content": system_template},
        {"role": "user", "content": user_template},
    ]
    prompt_template = prompt_templates.ChatPromptTemplate(
        template=messages_template, template_variables=template_variables
    )
    return prompt_template.populate(**kwargs)


def send_chat_message(client, model, messages, temperature=0.1, top_p=0.1):
    """
    Send a chat message to an LLM using `client`.
    `model` should be a valid model for the client.
    `message` should be a complete message or pre-formatted template.
    This function invokes the LLM using the chat protocol so that history
    is preserved.
    """
    return client.chat.completions.create(
        model=model, messages=messages, temperature=temperature, top_p=top_p
    )
