import yaml
from pathlib import Path
from openai import OpenAI

def load_model_config(model_name: str) -> dict:
    """Load model configuration from profiles.yaml"""
    config_path = Path(__file__).parent.parent / "configs" / "profiles.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config["models"][model_name]

def send_messages(messages):
    response = client.chat.completions.create(
        model=model_config["model"],
        messages=messages,
        tools=tools
    )
    return response.choices[0].message

# Load configuration from YAML
model_config = load_model_config("deepseek")

client = OpenAI(
    api_key=model_config["api_key"],
    base_url=model_config["base_url"],
)

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather of a location, the user should supply a location first.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    }
                },
                "required": ["location"]
            },
        }
    },
]

messages = [{"role": "user", "content": "How's the weather in Hangzhou, Zhejiang?"}]
message = send_messages(messages)
print(f"User>\t {messages[0]['content']}")

tool = message.tool_calls[0]
messages.append(message)

messages.append({"role": "tool", "tool_call_id": tool.id, "content": "24℃"})
message = send_messages(messages)
print(f"Model>\t {message.content}")