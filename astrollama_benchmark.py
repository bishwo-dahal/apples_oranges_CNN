import ollama
import time

MODEL_NAME="hf.co/UniverseTBD/astrollama.gguf:latest"

def query_llama(prompt):
    start_time = time.time()
    response = ollama.chat(model=MODEL_NAME, messages=[{'role': 'user', 'content': prompt}])
    end_time = time.time()
    elapsed_time = end_time - start_time
    return response['message']['content'], elapsed_time

if __name__ == "__main__":
    prompt = "Explain different types of stars in our universe."
    
    response, time_taken = query_llama(prompt)
    print("AstroLlama Response:")
    print(response)
    print(f"Time taken for API call: {time_taken:.2f} seconds")