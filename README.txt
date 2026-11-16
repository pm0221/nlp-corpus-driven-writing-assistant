NLP CORPUS-DRIVEN WRITING ASSISTANT

1. python -m pip install -r requirements.txt
2. python app.py
3. Open http://127.0.0.1:5000

There is NO predefined corpus in the UI. Paste your own corpus and train the model.
The supplied corpus becomes the model's vocabulary/probability source and the reference source for n-gram similarity.
The UI includes one meaningful visual: a bar chart comparing average probabilities under no smoothing, Add-k, and Laplace smoothing.
