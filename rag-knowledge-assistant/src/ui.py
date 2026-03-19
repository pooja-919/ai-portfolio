import gradio as gr
import requests

API_URL = "http://127.0.0.1:8000/ask"

def chat(question):
    try:
        response = requests.post(API_URL, json={"question": question})
        return response.json()["answer"]
    except Exception as e:
        return f"Error: {str(e)}"

interface = gr.Interface(
    fn=chat,
    inputs=gr.Textbox(placeholder="Ask something..."),
    outputs="text",
    title="RAG Assistant",
    description="Ask questions from your knowledge base"
)

if __name__ == "__main__":
    interface.launch()




