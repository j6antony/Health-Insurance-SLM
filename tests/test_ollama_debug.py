
import os
from ollama import Client

client = Client(host="http://localhost:11434")

response = client.chat(
    model=os.environ.get("SLM_MODEL_NAME", "qwen2.5:3b-instruct"),
    messages=[
        {
            "role": "user",
            "content": (
                'Classify "Hospital information" as a '
                'section heading. Return JSON containing '
                'only {"label": "section_heading"}.'
            )
        }
    ],
    format="json",
    think=False,
    options={
        "temperature": 0,
        "num_predict": 256
    }
)

print("Content:", repr(response.message.content))
print("Thinking:", repr(getattr(response.message, "thinking", None)))
print("Done reason:", response.done_reason)
print("Output tokens:", response.eval_count)
