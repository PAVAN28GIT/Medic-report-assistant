from google import genai

# Initialize Gemini client with your API key
client = genai.Client(api_key="AIzaSyCdJH3uSBwWKcyrKBlMTyI3NhJ1x4TK_ak")

# Base radiology report
base_text = """
A left chest wall dual lead AICD is present. Interval increase in size of a right pleural effusion with subjacent atelectasis. Additionally there are increasing streaky and patchy airspace opacities throughout the left mid and lower lung zones. A loculated left pleural effusion is again present. No pneumothorax identified. The size of the cardiac silhouette is enlarged but unchanged.
"""

# Prompt to generate expanded medical report
prompt = f"""
You are a medical expert assistant. Based on the following radiology report, please generate a comprehensive expansion with the following structure:

1. **Explanation of Complex Terms**: For each medical term in the report, provide a clear, concise explanation.
2. **Expanded Findings**: Rephrase the findings in professional and easy-to-understand medical language. Add relevant technical interpretation as needed.
3. **Impression**: Summarize the clinical interpretation of the findings (4–5 sentences).
4. **Diagnosis**: Suggest 4-5 possible diagnoses based on the findings.
5. **Recommendations**: Suggest 4-5 follow-up actions, tests, or treatments.

Radiology Report:
\"\"\"{base_text}\"\"\"

Generate a well-structured response under each section with 4–6 sentences where appropriate.
"""

# Generate expanded report
print("Generating expanded medical report...\n")
response = client.models.generate_content(
    model="gemini-2.0-flash",
    contents=prompt
)
expanded_report = response.text
print("\n--- Expanded Medical Report ---\n")
print(expanded_report)

# Start Q&A chatbot loop
print("\n--- Medical Q&A Chatbot ---")
print("You can now ask questions about the above medical report. Type 'exit' to end.\n")

while True:
    user_question = input("You: ")
    if user_question.lower() in ["exit", "quit"]:
        print("Goodbye!")
        break

    # Create a context-based prompt using the expanded report
    qa_prompt = f"""
    You are a helpful medical assistant. Based on the following expanded medical report, answer the user's question in a clear, medically accurate, and concise manner.

    Expanded Report:
    \"\"\"{expanded_report}\"\"\"

    User Question:
    \"\"\"{user_question}\"\"\"
    """

    answer = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=qa_prompt
    )
    print("Assistant:", answer.text)
