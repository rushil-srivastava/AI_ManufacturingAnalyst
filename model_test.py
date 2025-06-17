import time
from llama_cpp import Llama

# --- Configuration ---
# Adjust the path if your model file is located elsewhere
model_path = "/Users/rushilsrivastava/models/DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf"

# --- Model Loading ---
print(f"Loading GGUF model: {model_path}")
print("This might take a moment...")
load_start_time = time.time()

try:
    llm = Llama(
        model_path=model_path,
        n_ctx=4096,         # Context window size (adjust based on RAM and needs)
                            # Original model context is huge (131k)[7], but limited by RAM here.
        n_threads=None,     # Number of CPU threads to use (None = Llama.cpp detects)
        n_gpu_layers=0,     # Set to 0 to ensure CPU-only execution.
        verbose=False       # Set to True for more detailed Llama.cpp output
    )
    load_end_time = time.time()
    print(f"Model loaded successfully in {load_end_time - load_start_time:.2f} seconds.")

    # --- Prompting ---
    # Use the specific prompt format for this model [10]
    system_prompt = "You are a helpful AI assistant designed to answer questions about aerospace manufacturing based on provided context."
    user_question = "What is the most common root cause for part family 'XYZ-Component' to be delivered late?" # Example Q10 from your doc [1]

    # Construct the prompt using the required template
    prompt = f"<｜begin of sentence｜>{system_prompt}<｜User｜>{user_question}<｜Assistant｜>"

    print(f"\nUsing prompt template:\n{prompt}")
    print("\nGenerating response...")
    generation_start_time = time.time()

    # --- Generation ---
    output = llm(
        prompt,
        max_tokens=250,       # Max tokens to generate
        stop=["<｜", "User｜"], # Stop sequences to prevent rambling [10]
        echo=False           # Do not repeat the prompt in the output
    )

    generation_end_time = time.time()
    print(f"Response generated in {generation_end_time - generation_start_time:.2f} seconds.")

    # --- Output ---
    generated_text = output['choices'][0]['text'].strip()
    print("\n--- Generated Response ---")
    print(generated_text)

except FileNotFoundError:
    print(f"\n--- Error ---")
    print(f"Model file not found at path: {model_path}")
    print("Please ensure the path is correct and the GGUF file was downloaded.")
except Exception as e:
    print(f"\n--- Error ---")
    print(f"An unexpected error occurred: {e}")
    print("This could be due to installation issues, corrupted model file, or insufficient RAM.")

print("\n--- Script Complete ---")
