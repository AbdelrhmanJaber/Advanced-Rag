import ollama

# Test the connection and model generation
response = ollama.chat(
    model='llama3.2',  # Replace with your model name from 'ollama list'
    messages=[{'role': 'user', 'content': 'generate a simple story about poverty'}]
)

print(response['message']['content'])