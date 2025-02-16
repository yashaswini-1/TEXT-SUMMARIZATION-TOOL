from transformers import pipeline

def initialize_summarizer():
    """Initialize the summarization pipeline once to optimize performance."""
    return pipeline("summarization")

def summarize_text(summarizer, text, max_length=150, min_length=50):
    """Summarize the given text using the provided summarizer."""
    try:
        summary = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
        return summary[0]['summary_text']
    except Exception as e:
        return f"Error summarizing text: {e}"

def main():
    summarizer = initialize_summarizer()
    input_text = """Artificial Intelligence (AI) has revolutionized numerous industries, including healthcare, finance, and transportation. 
    AI-driven applications such as machine learning and natural language processing enable automation, improve efficiency, and enhance 
    decision-making processes. However, the widespread adoption of AI also raises ethical concerns related to data privacy, bias, and 
    job displacement. Governments and organizations are working towards establishing regulations to ensure responsible AI development. 
    The future of AI holds immense potential, but it requires careful oversight to balance innovation with ethical considerations."""
    
    summary = summarize_text(summarizer, input_text)
    print("Original Text:")
    print(input_text)
    print("\nSummarized Text:")
    print(summary)

if __name__ == "__main__":
    main()
