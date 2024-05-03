# NLP Challenge for Core42 AI Modelling Team

## Overview

This project is focused on developing and deploying a text classification model to detect hate speech in texts. A significant innovation in this project is the transition from sentence-level to comment-level classification, which effectively addresses the "relation" problem highlighted in prior studies. Additionally, the model goes beyond merely analyzing the hatefulness of individual words; it employs a transformer architecture to grasp the complex relationships between words within a context, thereby enhancing the accuracy of hate speech detection.

We have used the data available at [hate-speech-dataset](https://github.com/Vicomtech/hate-speech-dataset/tree/master?tab=readme-ov-file)

## Approach

### Data Preparation

The dataset comprises comments compiled from sentences originally extracted from a white supremacy forum, with each comment treated as a distinct document. Extensive Exploratory Data Analysis (EDA) was conducted to decipher the dataset's characteristics and to leverage the predictive power of cutting-edge NLP techniques. A novel approach adopted in this project, differing from previous methodologies, involves aggregating all sentences of a comment. This strategy is hypothesized to enable models equipped with attention mechanisms to more effectively capture the context and interrelations among words and sentences, which may collectively indicate hate speech.

### Models

Our modeling strategy includes the evaluation of three different models: TF-IDF, BCE Embedding, and a fine-tuned BCE Embedding, each paired with a logistic regression classifier. The BCE models utilizes a transformer-based architecture with an attention mechanism to discern the contextual relationships between words within comments. This method is designed to more accurately distinguish between harmful and benign uses of potentially hateful words (e.g., "ape" used derogatively versus in a non-hateful context).

### Deployment

The model is deployed as a REST API using FastAPI, running locally on port 8000. The API enables users to submit comments for classification and returns the model’s prediction regarding the presence of hate speech. Details of the deployment setup are provided in `app.py`.

### Usage

Users can interact with the API using a CURL command. Example usage:

Activate the virtual environment and install the dependencies:

```
poetry shell
poetry install
```
Launch the API:

```
uvicorn app:app
```

Submit a comment for classification:

```
curl -X POST http://localhost:8000/predict/ -H "Content-Type: application/json" -d '{"text": "Insert comment here.."}' 
```
