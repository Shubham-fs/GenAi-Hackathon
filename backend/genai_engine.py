import os
import base64
from io import BytesIO
from groq import Groq
from PIL import Image
from dotenv import load_dotenv

load_dotenv()

class GenAIEngine:
    def __init__(self, model_name="meta-llama/llama-4-scout-17b-16e-instruct"):
        """
        Initialize the Groq model for multimodal tasks.
        Using the recommended Llama 3.2 11B Vision model.
        """
        self.model_name = model_name
        self.client = None
    
    def _get_client(self):
        if not self.client:
            api_key = os.environ.get("GROQ_API_KEY", "")
            if api_key and api_key != "your_api_key_here":
                self.client = Groq(api_key=api_key)
        return self.client

    def _encode_image(self, image: Image.Image) -> str:
        buffered = BytesIO()
        # Convert to RGB if not already
        if image.mode != "RGB":
            image = image.convert("RGB")
        image.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

    def _generate(self, prompt: str, image: Image.Image = None) -> str:
        client = self._get_client()
        if not client:
            return "Error: Groq API key not found. Please add it to your .env file or the sidebar."
        
        try:
            content = [{"type": "text", "text": prompt}]
            
            if image:
                base64_img = self._encode_image(image)
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_img}"
                    }
                })
                
            response = client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "user",
                        "content": content
                    }
                ],
                temperature=0.7,
                max_tokens=1024
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error connecting to Groq AI: {str(e)}"

    def generate_explanation(self, image: Image.Image, regions_context: str, complexity_level: str = "Normal") -> str:
        level_instructions = {
            "Explain Simply (Like I'm 10)": "Explain this using very simple words, easy analogies, and short sentences as if you were speaking to a 10-year-old child.",
            "Explain Normally": "Provide a clear, educational explanation suitable for a high school or college student.",
            "Explain Technically": "Provide a highly technical, detailed, and scientifically accurate overview suitable for an expert or medical/engineering professional."
        }
        
        instruction = level_instructions.get(complexity_level, level_instructions["Explain Normally"])
        
        prompt = f"""
        Act as an expert AI Visual Tutor. I am providing you with an educational diagram. 
        
        Context from Semantic Segmentation:
        {regions_context}
        
        Task:
        Provide a structured explanation of the entire diagram and how its parts connect and work together.
        
        Instruction:
        {instruction}
        
        Format your response clearly using markdown headings and bullet points where useful.
        """
        return self._generate(prompt, image)

    def generate_flashcards(self, image: Image.Image, regions_context: str) -> str:
        prompt = f"""
        Act as an AI Flashcard Creator for study materials.
        
        Context from Semantic Segmentation:
        {regions_context}
        
        Task:
        Analyze the provided diagram and the detected regions, then generate exactly 5 flashcards to help a student memorize key concepts and functions shown in the diagram.
        
        Format:
        Use the following markdown format:
        
        **Flashcard 1**
        *Front:* (Question)
        *Back:* (Answer)
        
        ...and so on up to 5.
        """
        return self._generate(prompt, image)
        
    def generate_quiz(self, image: Image.Image, regions_context: str) -> str:
        prompt = f"""
        Act as an AI Quiz Generator for study materials.
        
        Context from Semantic Segmentation:
        {regions_context}
        
        Task:
        Based on the diagram provided, create 3 multiple-choice questions to test the student's understanding.
        Ensure the questions are actually answered by the relationships shown in the visual diagram.
        
        Format:
        For each question:
        **Question 1:** [Question text]
        A) Option A
        B) Option B
        C) Option C
        D) Option D
        
        **Correct Answer:** [Letter] - [Brief explanation of why]
        """
        return self._generate(prompt, image)

    def generate_poster(self, image: Image.Image, regions_context: str) -> str:
        prompt = f"""
        Act as an AI Poster Designer.
        
        Context from Semantic Segmentation:
        {regions_context}
        
        Task:
        Create the text content for a simplified learning poster based on the provided diagram.
        The layout should summarize the core concept visually within a text format.
        
        Include:
        # 📌 [Catchy Title Summarizing the Concept]
        
        ## Key Components
        (List the 3-4 most important parts and their 1-sentence function)
        
        ## The Big Picture
        (A 2-sentence summary of the overall process or system)
        
        Make it short, catchy, and highly readable!
        """
        return self._generate(prompt, image)

    def answer_question(self, image: Image.Image, regions_context: str, user_question: str) -> str:
        prompt = f"""
        Act as an AI Tutor answering a student's specific question about a diagram.
        
        Context from Semantic Segmentation:
        {regions_context}
        
        Student Question: 
        "{user_question}"
        
        Task:
        Answer the question directly, conversational, and accurately based purely on the provided image and segmentation context. Keep the answer to a concise paragraph.
        """
        return self._generate(prompt, image)
