import streamlit as st
import PyPDF2
import os
from openai import OpenAI
from dotenv import load_dotenv


# load_dotenv()


client = OpenAI(
    base_url="https://api.aimlapi.com/v1",  
    # api_key=os.getenv("DEEPSEEK_API_KEY"),  
    api_key=st.secrets["DEEPSEEK_API_KEY"],
)

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    """
    Extracts text from a PDF file.
    """
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text


def recognize_disease(medical_report_text, temperature, blood_pressure, cholesterol_level):
    """
    Recognizes disease using DeepSeek API.
    """
    prompt = (
        f"You are a medical AI assistant who specializes in cardiovascular diseases. "
        f"Here is the patient's medical report:\n\n{medical_report_text}\n\n"
        f"Additional patient data:\n"
        f"- Temperature: {temperature} °C\n"
        f"- Blood Pressure: {blood_pressure} mmHg\n"
        f"- Cholesterol Level: {cholesterol_level} mg/dL\n\n"
        f"Based on the medical report and the additional data, what is the likely cardiovascular disease? "
        f"Provide a concise diagnosis and explain your reasoning in simple terms for the patient."
    )
    
    # Call DeepSeek API with increased max_tokens
    response = client.chat.completions.create(
        model="deepseek/deepseek-r1",  # Replace with the correct model name
        messages=[
            {"role": "system", "content": "You are a medical AI assistant who specializes in cardiovascular diseases."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=1000,  # Increased token limit for longer responses
    )
    return response.choices[0].message.content

# Function to generate recommendations using DeepSeek API
def generate_recommendations(disease, medical_report_text):
    """
    Generates recommendations using DeepSeek API.
    """
    prompt = (
        f"The patient has been diagnosed with {disease}. "
        f"Here is their medical report:\n\n{medical_report_text}\n\n"
        f"Please provide the following in a clear, structured, and patient-friendly format:\n"
        f"1. **Diet Plan**: Suggest a personalized diet plan with specific foods to eat and avoid.\n"
        f"2. **Exercise**: Recommend safe and effective exercises.\n"
        f"3. **Medications**: List recommended medications (if any) and their purpose.\n"
        f"4. **Prevention Tips**: Provide actionable tips for preventing complications.\n"
        f"Use bullet points and simple language so the patient can easily understand."
    )
    
    # Call DeepSeek API with increased max_tokens
    response = client.chat.completions.create(
        model="deepseek/deepseek-r1",  
        messages=[
            {"role": "system", "content": "You are a medical AI assistant who specializes in cardiovascular diseases."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=3000,  
    )
    return response.choices[0].message.content

# Main Streamlit App
def main():
    st.set_page_config(page_title="CardioCare", page_icon="❤️", layout="wide")
    st.title("CardioCare AI: Smart Cardiovascular Diagnosis & Personalized Treatment")
    st.write("Welcome to **CardioCare**! Please enter your details below to get a personalized diagnosis and treatment plan.")

    # Input Fields
    st.sidebar.header("Patient Information")
    temperature = st.sidebar.number_input("Body Temperature (°C)", min_value=30.0, max_value=45.0, value=37.0)
    blood_pressure = st.sidebar.number_input("Blood Pressure (mmHg)", min_value=0, max_value=300, value=120)
    cholesterol_level = st.sidebar.number_input("Cholesterol Level (mg/dL)", min_value=0, max_value=500, value=200)
    medical_report_pdf = st.sidebar.file_uploader("Upload Medical Test Report (PDF)", type="pdf")

    # Submit Button
    if st.sidebar.button("Get Recommendations"):
        if medical_report_pdf is not None:
            # Extract text from PDF
            medical_report_text = extract_text_from_pdf(medical_report_pdf)

            # Recognize Disease
            with st.spinner("Analyzing your data..."):
                recognized_disease = recognize_disease(medical_report_text, temperature, blood_pressure, cholesterol_level)

            # Generate Recommendations
            with st.spinner("Generating personalized recommendations..."):
                recommendations = generate_recommendations(recognized_disease, medical_report_text)

            # Display Output
            st.header("Diagnosis and Recommendations")
            st.subheader("1. Recognized Disease")
            st.success(recognized_disease)

            st.subheader("2. Recommendations")
            st.markdown(recommendations)  # Use markdown for better formatting
        else:
            st.error("Please upload a medical test report in PDF format.")

    # Footer
    st.sidebar.markdown("---")
    st.sidebar.write("Developed by Kalam Tech Team")
    st.sidebar.write("Powered by **DeepSeek**")

if __name__ == "__main__":
    main()
