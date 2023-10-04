# FindMyMovie | Intelligent Movie Search

A project aimed at enhancing the user movie search system in Russian.

### Phase-2 | Team Project

## 🦸‍♂️ Team
1. [Salman Chakaev](https://github.com/veidlink)
2. [Grisha Rzhischev](https://github.com/Rzhischev)
3. [Dmitri Vanakin](https://github.com/cobalt1705)

## 🎯 Task
Development of an application using Streamlit.
The service is deployed on [HuggingFace Spaces](https://huggingface.co/spaces/veidlink/find_my_movie_hf).

## 🚂 Model
The application operates on the BERT model - [rubert-tiny2](https://huggingface.co/cointegrated/rubert-tiny2).

## 📝 Workflow
1. We have scraped 12,000 movies from the mail.ru catalog. The information used for recommendations includes the movie description from the movie page and editorial reviews.
2. Bert encodes the description+review for each movie into a vector.
3. The user enters the movie description, which is also passed through BERT, yielding encoded information in vector form.
4. Using [faiss](https://github.com/facebookresearch/faiss), based on the Euclidean distance between the user's description and the movies from the mail.ru catalog, a selected number of predictions with the highest similarity is displayed.

## 📚 Libraries 

```typescript
import numpy
import pandas 
import faiss
import torch
import joblib
import streamlit 
from transformers import AutoTokenizer, AutoModel
```

## 📚 Guide 
### How to run locally?

1. To create a Python virtual environment for running the code, enter:

    ``python3 -m venv my-env``.

2. Activate the new environment:

    * Windows: ```my-env\Scripts\activate.bat```
    * macOS and Linux: ```source my-env/bin/activate```

3. Install all dependencies from the *requirements.txt* file:

    ``pip install -r requirements.txt``.
