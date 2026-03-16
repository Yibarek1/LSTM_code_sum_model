\# CSCI 455/555: GenAI for SD - Assignment 2

\*\*Summarizing code via LSTM models\*\*



This repository contains the implementation of an LSTM-based model designed to generate natural language summaries for Java methods. The model utilizes CodeT5+ pretrained embeddings and is evaluated using BLEU, METEOR, BERTScore, and the SIDE metric.



\## 1. Data Sources and Pre-processing

The dataset consists of Java code-summary pairs mined from public GitHub repositories. 

\* \*\*Data Split:\*\* The dataset contains \~50,000 code-summary pairs for training and 1,000 samples for the validation set. The final evaluation is performed on a held-out test set of 1,000 pairs provided by the instructor.

\* \*\*Pre-processing:\*\* Each Java method was flattened into a single whitespace-normalized line, and the corresponding natural language summaries were lowercased. The pairs were saved as `.txt` files with one sample per line.

\* \*\*Tokenization \& Embeddings:\*\* The `get\_codet5\_embeddings.py` script was used to tokenize the text and extract the CodeT5+ pretrained embedding matrix. The maximum sequence length was empirically set to 256 for Java methods and 512 for the Javadoc summaries based on sequence length percentiles.



\## 2. How to Install Dependencies and Reproduce

To run the code and reproduce the results, follow these steps:



\*\*Step 1: Setup the Environment\*\*

Create and activate a virtual environment, then install the required dependencies:

```bash

python3 -m venv venv

source venv/bin/activate

pip install -r requirements.txt

```

\*Note: Additional evaluation libraries (`sacrebleu`, `evaluate`, `sentence-transformers`) are required and can be installed via the notebook cells.\*



\*\*Step 2: Download the SIDE Metric Model\*\*

To calculate the SIDE metric, you must download the custom triplet-loss Sentence Transformer model from the repository provided in the assignment instructions. Update the `SIDE\_CHECKPOINT` path in the evaluation cell of the notebook to point to this downloaded folder.



\*\*Step 3: Generate Embeddings\*\*

Run the provided embedding script on the training, validation, and test datasets:

```bash

python get\_codet5\_embeddings.py --input train\_code.txt --output train\_code.pt --max\_length 256

python get\_codet5\_embeddings.py --input train\_summary.txt --output train\_summary.pt --max\_length 512

\# Repeat for val\_\*.txt and test\_\*.txt

```



\*\*Step 4: Run the Notebook\*\*

Open `assignment-2-LSTM.ipynb` and run it top-to-bottom. The notebook will load the `.pt` files, train the LSTM with early stopping based on Validation BLEU-1, and evaluate the final test set.



\## 3. Where Outputs are Written

\* \*\*Embeddings:\*\* The `.pt` files containing token IDs and the embedding matrix are saved in the root directory alongside the `.txt` files.

\* \*\*Model Checkpoints:\*\* The best-performing model weights (saved via the early stopping mechanism) are written to the `checkpoints/` directory (e.g., `checkpoints/lstm\_summarization\_model.pt`).

\* \*\*Predictions and Evaluation:\*\* The final generated textual predictions and the computed metrics (BLEU-1,2,3,4, METEOR, BERTScore, and SIDE) are output directly within the final cells of the Jupyter Notebook.

