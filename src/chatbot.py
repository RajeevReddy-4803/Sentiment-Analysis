from transformers import pipeline

chatbot = pipeline("text-generation", model="distilgpt2")

def generate_response(prompt, feedback_summary):
    """
    Simple chatbot response using summarized feedback as context.
    """
    context = f"Customer feedback summary: {feedback_summary}\n\nUser query: {prompt}\nAssistant response:"
    response = chatbot(context, max_length=150, num_return_sequences=1, do_sample=True, temperature=0.7)
    return response[0]['generated_text'].split("Assistant response:")[-1].strip()
