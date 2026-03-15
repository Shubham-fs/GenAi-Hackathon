# AI Visual Tutor - Smart Study Material Generator 🧠📚

**AI Visual Tutor** is an interactive, AI-powered educational tool designed to instantly convert complex diagram images into personalized study materials. 

Built for hackathons and ed-tech innovation, this system solves a core problem in education: helping students truly *understand* visual science and engineering diagrams rather than just memorizing them.

## 🚀 Key Features

The pipeline accepts an image, runs **Semantic Segmentation** to map out components, and uses **Generative AI** (Llama 4 Vision via Groq) to generate:
- **Diagram Explanations:** Explaining relationships at three complexity levels (Simple, Normal, Technical).
- **Study Flashcards:** Auto-generated question & answer cards to test recall.
- **Practice Quizzes:** Context-aware multiple-choice questions.
- **Summary Posters:** High-level visual summaries of the diagram's ecosystem.
- **Interactive Q&A:** A built-in tutor that answers specific student questions about the uploaded image.

## 🛠️ Technology Stack
- **Frontend:** Streamlit 
- **Computer Vision:** Hugging Face `nvidia/segformer-b0` (Semantic Segmentation)
- **Generative AI:** Groq API (`meta-llama/llama-4-scout-17b-16e-instruct`)
- **Languages/Tools:** Python, PyTorch, OpenCV, Pillow

## ⚙️ How to Run Locally
1. Clone the repository.
2. Install dependencies: `pip install -r requirements.txt`
3. Create a `.env` file and add your `GROQ_API_KEY`.
4. Start the frontend: `python -m streamlit run frontend/app.py`
