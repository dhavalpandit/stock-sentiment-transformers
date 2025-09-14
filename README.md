# stock-sentiment-transformers
Stock &amp; Sentiment Analysis Using Transformers – Fine-tuned DistilBERT, FinBERT, RoBERTa, and DistilRoBERTa on 6K+ tweets, achieving up to 89% accuracy for financial sentiment classification with real-time deployment focus.


# 📊 Stock & Sentiment Analysis Using Transformers

This project applies **transformer-based NLP models** to classify financial sentiment from stock-related tweets.  
We fine-tuned **DistilBERT, FinBERT, RoBERTa, and DistilRoBERTa** using a dataset of **6,000+ tweets** to detect *positive* and *negative* sentiments, benchmarked for **accuracy, speed, and model size**.  

---

## 🚀 Features
- End-to-end **NLP pipeline**: preprocessing → tokenization → model training → evaluation → inference  
- Fine-tuning of multiple Hugging Face transformer models  
- Achieved **up to 89% accuracy** with FinBERT, and fastest inference with DistilRoBERTa  
- **Benchmark comparison** of models across accuracy, speed, and deployment readiness  
- Scalable framework for **real-time financial sentiment prediction**  

---

## 🛠️ Tech Stack
- **Python 3.9+**  
- **Transformers (Hugging Face)**  
- **PyTorch**  
- **scikit-learn, pandas, numpy, matplotlib**  
- **Streamlit** (for visualization & interactive dashboard)  

---

## 📂 Project Structure
```
.
├── configs/              # Config files (paths, hyperparameters)
├── data/                 # Raw & processed data
├── models/               # Trained models (checkpoints)
├── src/
│   ├── preprocess.py     # Text cleaning & preprocessing
│   ├── train.py          # Model training & saving
│   ├── evaluate.py       # Evaluation metrics
│   └── infer.py          # Inference on new text
├── requirements.txt      # Dependencies
├── README.md             # Project documentation
└── LICENSE               # MIT License
```

---

## ⚡ Quickstart
1. Clone the repository:
   ```bash
   git clone https://github.com/<your-username>/stock-sentiment-transformers.git
   cd stock-sentiment-transformers
   ```

2. Create environment & install dependencies:
   ```bash
   python3 -m venv venv && source venv/bin/activate
   pip install -r requirements.txt
   ```

3. Prepare dataset:
   - Add your CSV file under `data/raw/tweets.csv`  
   - Format:  
     ```
     text,label
     "AAPL is rising after earnings",positive
     "TSLA stock crash incoming",negative
     ```

4. Run pipeline:
   ```bash
   python src/preprocess.py --config configs/config.yaml
   python src/train.py --config configs/config.yaml
   python src/evaluate.py --config configs/config.yaml
   ```

5. Inference:
   ```bash
   python src/infer.py --text "Apple stock is performing well"
   ```

---

## 📊 Results
- **FinBERT** → Highest accuracy (**89%**) on financial text  
- **DistilRoBERTa** → Best trade-off (accuracy ~88% with 35% faster inference)  
- **DistilBERT & RoBERTa** → Balanced models for general NLP tasks  

---

## 📜 License
This project is licensed under the [MIT License](LICENSE).  

---

## 🙌 Acknowledgments
- Hugging Face 🤗 for pretrained models  
- **UTA ML Project Team (Group 5)** – *Dhaval Pandit, Arth Patel, Karan Patel*  
