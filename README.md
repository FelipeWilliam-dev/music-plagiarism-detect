# 🎶 Identifying Plagiarism in Musical Patterns Using Deep Learning

Music, as a form of artistic expression, is constantly subject to influence and reinterpretation. However, the line between inspiration and plagiarism often becomes blurred, leading to legal and ethical disputes in the music industry. With the advancement of Artificial Intelligence (AI) and Deep Learning, there is now the possibility to automate the analysis of musical patterns, identifying similarities that may constitute unauthorized copying.

## 🛠️ How to Use

### ✅ Requirements

- Python **3.11**
- `ffmpeg` installed and available in PATH (required for `pydub`)
- Virtual environment configured with:

```bash
pip install -r requirements.txt
``` 

### 🧰 Dependencies
```bash
librosa
numpy
scipy
pydub
setuptools
``` 

### ⚙️ Setup
#### 1. Clone the repository
```bash
git clone https://github.com/FelipeWilliam-dev/music-plagiarism-detect

cd music-plagiarism-detect
``` 
#### 2. Create and activate the virtual environment

```bash
python3.11 -m venv venv
venv\Scripts\activate  # on Windows
source venv/bin/activate  # on Linux/macOS
``` 

#### 3. Install the dependencies
```bash
pip install -r requirements.txt
``` 

## 🎵 How to Compare Two Songs

1. Place your `.mp3` songs in the `Musicas/` folder (e.g., `Musicas/Song1.mp3`, `Musicas/Song2.mp3`)
2. Run the script:

```bash
python main.py
```

3. The program will display a similarity score between the two songs in the terminal, for example:

```
Song A and B have a similarity of: 0.82
```

---

## 🧠 How It Works

- Automatic `.mp3` to `.wav` conversion with `pydub`
- Feature extraction using `librosa`:
  - MFCC (Mel-frequency cepstral coefficients)
  - Chroma (harmonic representation)
- Alignment and distance calculation using `DTW` (Dynamic Time Warping)
- Generation of a standardized similarity metric

---

## 📁 Project Structure

```
music-plagiarism-detect/
├── Musicas/
│   ├── Music1.mp3
│   └── Music2.mp3
├── main.py
├── requirements.txt
└── README.md
```

---

## 📌 Notes

- This project is a **proof of concept**, not a legal tool.
- Future enhancements will include deep neural networks for musical embedding extraction.

---

## 📜 License

This project is distributed under the MIT License. See the `LICENSE` file for more details.
