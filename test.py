from openai import OpenAI

client = OpenAI()

response = client.responses.create(
    model="gpt-4.1-mini",
    input="Return JSON {\"test\": true}"
)

print(response.output_text)