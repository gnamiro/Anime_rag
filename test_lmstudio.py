from openai import OpenAI

client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

resp = client.chat.completions.create(
    model="phi-3-mini-4k-instruct",  # if this errors, use the exact name shown in LM Studio
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Say hello in one sentence."},
    ],
    temperature=0.2,
)

print(resp.choices[0].message.content)