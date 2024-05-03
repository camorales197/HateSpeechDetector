from sentence_transformers import SentenceTransformer
from cleantext.sklearn import CleanTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from joblib import load

class TextPredictor:
    """Class to handle text prediction using a fine-tuned SentenceTransformer and a logistic regression classifier."""

    def __init__(self):
        """Initialize the TextPredictor with model, pipeline, and cleaner configurations."""
        # Load your custom model and tokenizer
        model_save_path = "CarlosMorales/HateSpeechDetector"
        matryoshka_dim = 64
        self.model = SentenceTransformer(model_save_path, truncate_dim=matryoshka_dim)
        self.classifier = load('models/logistic_regression_classifier.joblib')
        self.cleaner = CleanTransformer(
        )

    def text_to_finetuned_embedding(self, text):
        """Convert text to embeddings using the fine-tuned model.

        Args:
            text (str): The text to convert to embeddings.

        Returns:
            tensor: The tensor representation of the text embeddings.
        """
        return self.model.encode(text, convert_to_tensor=True).cpu()

    def predict_text(self, text: str):
        """Preprocess and classify the text.

        Args:
            text (str): The text to predict.

        Returns:
            str: The prediction result.
        """
        cleaned_text = self.cleaner.transform([text])[0]
        embeddings = self.text_to_finetuned_embedding(cleaned_text)
        prediction = int(self.classifier.predict([embeddings]))
        if prediction == 1:
            return "HateSpeech"
        elif prediction == 0:
            return "NonHateSpeech"
        else:
            raise ValueError("Invalid prediction value.")
