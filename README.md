# NLP Corpus-Driven Writing Assistant

A web-based NLP application that learns language patterns from a corpus supplied by the user. After training, it can suggest likely next words, offer spelling candidates, analyse sentence probabilities, estimate perplexity, and compare text with the supplied corpus using n-gram overlap.

**Author:** Prayag Maru

---

## Features

- **User-provided corpus:** There is no predefined corpus. Paste your own text and train the model.
- **N-gram language model:** Builds unigram, bigram, trigram, 4-gram, and 5-gram frequency counts from the supplied corpus.
- **Next-word suggestions:** Ranks up to five possible next words based on the preceding context.
- **Spelling suggestions:** If the final typed word is not in the trained vocabulary, suggests similar corpus words using edit distance and language-model probability.
- **Sentence analysis:** Calculates transition probabilities for the words in the entered sentence.
- **Smoothing comparison:** Compares probabilities with no smoothing, Add-k smoothing, and Laplace smoothing. The Add-k value used is 0.5; Laplace uses 1.0.
- **Perplexity:** Estimates how predictable the entered sentence is according to the trained language model.
- **Corpus similarity indicator:** Compares distinct trigrams in the entered text with trigrams in the supplied corpus and reports the overlap.
- **Visual comparison:** Displays a bar chart comparing average bigram probabilities under the three smoothing approaches.
- **Conclusions:** Produces a short interpretation of the probability, similarity, and perplexity results.

> The similarity result is an n-gram overlap indicator, not a definitive plagiarism detector. High overlap should be reviewed in context.

---

## How It Works

### 1. Train with your corpus
Paste text into the **Provide the Corpus** area and select **Train Model**. The application tokenizes the text, converts tokens to lowercase, and builds n-gram counts from one word up to five words.

The corpus must contain at least eight words. Its vocabulary and word-pattern counts become the source used by the application for prediction and analysis.

### 2. Predict the next word
Enter text as a prompt. The application uses the most recent context and the trained n-gram counts to rank up to five next-word suggestions. If the final word is not in the vocabulary, it also generates spelling candidates.

### 3. Analyse a sentence
Enter a sentence and select **Analyze Sentence**. The application returns:
- A word-transition probability comparison.
- Perplexity, when the sentence contains at least two tokens.
- Trigram overlap with the supplied corpus.
- Matched phrases (where available).
- A short automatically generated conclusion.
- A bar chart comparing average probabilities for no smoothing, Add-k, and Laplace smoothing.

### Probability and similarity methods
- **No smoothing:** Uses the observed bigram frequency divided by the context frequency. Unseen bigrams receive probability zero.
- **Add-k smoothing:** Adds 0.5 to the observed count and adjusts the denominator using the vocabulary size.
- **Laplace smoothing:** Uses Add-k smoothing with k = 1.
- **Perplexity:** Calculated from the model's smoothed next-word probabilities. Lower perplexity means the model finds the sentence more predictable.
- **Trigram overlap:** Measures the percentage of distinct trigrams in the entered text that also occur in the trained corpus.

---

## Technologies Used

- Python
- Flask
- HTML templates (served by Flask)
- JavaScript and chart visualization in the web interface
- Python standard library modules: `math`, `re`, and `collections`

## Project Structure

Keep the files and folders arranged as follows:

```text
NLP_Corpus_Driven_Writing_Assistant/
├── app.py
├── requirements.txt
├── README.md
└── templates/
    └── index.html
```

The `templates/index.html` file is required because Flask uses it to render the application's home page. If you received the project as an archive, extract the archive first and run the commands from the extracted project folder.

---

## Requirements

- Python 3.10 or newer recommended
- pip
- A web browser
- An internet connection may be needed to install Flask, depending on your environment.

## Installation and Setup (Windows)

1. Download and extract the project archive.
2. Open the extracted project folder in File Explorer.
3. Click the folder address bar, type `cmd`, and press Enter to open Command Prompt in that folder.
4. Create a virtual environment:

   ```bash
   py -m venv .venv
   ```

5. Activate the virtual environment:

   ```bash
   .venv\Scripts\activate
   ```

6. Upgrade pip and install dependencies:

   ```bash
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

## Run the Application

From the project folder, with the virtual environment activated, run:

```bash
python app.py
```

When Flask starts, open this address in your browser:

```text
http://127.0.0.1:5000
```

To stop the local server, return to the terminal and press `Ctrl + C`.

### macOS / Linux

The same setup can be performed with:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python app.py
```

Then open `http://127.0.0.1:5000` in a browser.

---

## Using the Application

1. Open the application in your browser.
2. Paste your own text into **Provide the Corpus**.
3. Select **Train Model** and wait for the training confirmation.
4. Enter a short prompt to get next-word suggestions.
5. Enter a sentence and select **Analyze Sentence** to view probabilities, perplexity, corpus similarity, conclusions, and the chart.
6. If you want to experiment with different language patterns, provide another corpus and train the model again.

Example corpus:

```text
The student goes to college every day.
The student studies computer science.
The teacher teaches computer science.
Students learn algorithms and data structures.
```

Example prompt after training:

```text
The student studies
```

The suggestions depend on the corpus you provide; results will differ when the corpus changes.

---

## API Endpoints

The Flask application exposes the following routes:

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/` | Loads the web interface. |
| `GET` | `/api/health` | Returns whether a model is trained and the token and vocabulary counts. |
| `POST` | `/api/train` | Trains the model using a JSON `corpus` value. |
| `POST` | `/api/predict` | Returns next-word predictions and spelling candidates for JSON `text`. |
| `POST` | `/api/analyze` | Returns probability rows, perplexity, trigram similarity, and conclusions for JSON `text`. |

### Example request bodies

Train:

```json
{
  "corpus": "The student studies computer science. The teacher teaches computer science."
}
```

Predict:

```json
{
  "text": "The student studies"
}
```

Analyze:

```json
{
  "text": "The student studies computer science."
}
```

---

## Important Notes and Limitations

- The application does not include a built-in or default corpus. You must supply and train it before prediction or analysis.
- The model is stored in the Flask process's memory. Training is lost when the application process stops or restarts.
- The language model is based on word-frequency n-grams; it is not a neural network or a generative large language model.
- Next-word suggestions are limited to words found in the trained corpus vocabulary.
- Spelling suggestions are generated from the trained vocabulary and are not a full spell-checking system.
- The similarity feature compares distinct trigrams. It can identify phrase overlap with the supplied corpus, but cannot establish plagiarism or authorship.
- Perplexity and probability results depend on the corpus and the text being analysed. They should be interpreted as outputs of this model, not as universal language-quality scores.
- The Flask app is configured with `debug=True` for local development. Do not expose the development server directly to the public internet or use it as a production deployment without appropriate security and deployment changes.

## Troubleshooting

**`ModuleNotFoundError: No module named 'flask'`**

Activate your virtual environment and run:

```bash
python -m pip install -r requirements.txt
```

**The page cannot be loaded or Flask reports a missing template**

Check that `templates/index.html` exists inside the project folder and that you start `python app.py` from the folder containing `app.py`.

**Prediction says to train a model first**

Paste a corpus containing at least eight words and select **Train Model** before requesting predictions or sentence analysis.

**Port 5000 is already in use**

Stop the other application using that port, or configure this Flask application to use another available port.

---

## Author

**Prayag Maru**
