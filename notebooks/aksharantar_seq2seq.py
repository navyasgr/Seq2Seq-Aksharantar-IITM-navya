# ═══════════════════════════════════════════════════════════════════════════════
# 🎯 AKSHARANTAR TRANSLITERATION: Complete IIT Madras Submission
# ═══════════════════════════════════════════════════════════════════════════════
# Sequence-to-Sequence Model with Bahdanau Attention
# Roman Script → Devanagari Transliteration
# ═══════════════════════════════════════════════════════════════════════════════

# MARKDOWN CELL 1
"""
# 🎯 Aksharantar Transliteration: Roman → Devanagari

**IIT Madras Deep Learning Technical Aptitude Assignment**

---

## 📚 Step 1: Introduction and Problem Understanding

### 🔍 Problem Statement

The **Aksharantar dataset** (by AI4Bharat) contains word pairs of the form:
```
(romanized_word, native_script_word)
Example: ("ajanabee", "अजनबी")
```

Our task is to build a neural sequence-to-sequence model that learns:
```
f(x) = y
```
where:
- **x** = sequence of Latin characters (e.g., "ghar")
- **y** = sequence of Devanagari characters (e.g., "घर")

This is a **character-level translation problem**, similar to full-scale machine translation but operating at the character level.

---

### 🧠 Solution Approach

We implement a **Recurrent Neural Network (RNN) based Encoder-Decoder architecture** with **Bahdanau Attention**:

#### 1️⃣ **Encoder (Bidirectional LSTM)**
- Processes input Latin characters from both directions
- Captures contextual information effectively
- Outputs hidden states for each time step

#### 2️⃣ **Attention Mechanism (Bahdanau)**
- Computes context-aware weighted representations
- Allows decoder to focus on relevant encoder states dynamically
- **Formula**: `score(hₜ, h̄ₛ) = vᵀ tanh(W₁hₜ + W₂h̄ₛ)`

#### 3️⃣ **Decoder (Unidirectional LSTM)**
- Generates Devanagari characters sequentially
- Uses attention context at each step
- Outputs probability distribution over target vocabulary

---

### 🎯 Design Decisions

| Component | Choice | Rationale |
|-----------|--------|-----------|
| **Cell Type** | LSTM | Better gradient flow than vanilla RNN, captures long-range dependencies |
| **Attention** | Bahdanau | Dynamic focus on input positions, improves alignment quality |
| **Encoder** | Bidirectional | Captures context from both directions for richer representations |
| **Optimizer** | AdamW | Decoupled weight decay, adaptive learning rates, faster convergence |
| **Scheduler** | ReduceLROnPlateau | Reduces LR when validation plateaus, helps escape local minima |
| **Precision** | Mixed (AMP) | 2x faster training with float16, maintains float32 stability |

---

### 📊 Model Flexibility

Our implementation allows configurable:
- ✅ Embedding dimension (default: 256)
- ✅ Hidden state size (default: 512)
- ✅ Cell type (LSTM/GRU toggle)
- ✅ Number of encoder/decoder layers
- ✅ Dropout rate for regularization

---

### 🎓 Learning Objectives

1. Understand sequence-to-sequence architectures
2. Implement attention mechanisms from scratch
3. Apply modern deep learning training techniques
4. Analyze computational complexity theoretically
5. Follow software engineering best practices

---

Let's begin the implementation! 🚀
"""

# CODE CELL 1.1 - Initial Setup
# Install required packages (if not already installed)
!pip install -q torch torchvision torchaudio

# Import all required libraries
import os
import re
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.notebook import tqdm
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# PyTorch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torch.cuda.amp import autocast, GradScaler

print("✅ All libraries imported successfully!")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

# CODE CELL 1.2 - Set Random Seeds
# Set random seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n🖥️  Device Configuration:")
print(f"   Using: {device}")
if torch.cuda.is_available():
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    print("   ⚠️  Warning: Running on CPU. Training will be slow!")

# CODE CELL 1.3 - Visualization Setup
# Set up plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10

print("✅ Visualization settings configured!")

# ═══════════════════════════════════════════════════════════════════════════════

# MARKDOWN CELL 2
"""
---

## 📂 Step 2: Dataset Preparation and Preprocessing

### 📥 Dataset Structure

The Aksharantar dataset is organized as:
```
aksharantar_sampled/
    ├── hin/                    # Hindi language subset
    │   ├── hin_train.csv
    │   ├── hin_valid.csv
    │   └── hin_test.csv
    └── [other languages...]
```

Each CSV file contains **two columns without headers**:
- **Column 0**: Native script word (Devanagari)
- **Column 1**: Romanized transliteration (Latin)

**Note**: We reverse this mapping since our task is **Latin → Devanagari**

---

### 🧹 Preprocessing Pipeline

1. **Text Normalization**: Remove extra whitespace, convert Latin to lowercase
2. **Special Tokens**: Add `<SOS>` (start), `<EOS>` (end), `<PAD>` (padding)
3. **Character Tokenization**: Split words into individual characters
4. **Vocabulary Building**: Create char→index mappings for both scripts
5. **Length Filtering**: Remove extremely long sequences (>50 chars)

---

### 📊 Why These Choices?

- **Lowercase Latin**: Reduces vocabulary size, improves generalization
- **Character-level**: Handles unseen word forms better than word-level
- **Length filtering**: Improves training stability and speed
- **Special tokens**: Enable proper sequence generation with start/end markers

---
"""

# CODE CELL 2.1 - Mount Google Drive
from google.colab import drive

# Mount Google Drive
drive.mount('/content/drive')

# CODE CELL 2.1B - Alternative: Download Dataset Directly (if not in Drive)
# Uncomment this section if you want to download directly instead of using Drive

"""
# Download from Google Drive link (alternative method)
print("📥 Downloading dataset from Google Drive...")

# Install gdown if needed
!pip install -q gdown

# Download the dataset
import gdown
file_id = "YOUR_GOOGLE_DRIVE_FILE_ID"  # Replace with actual file ID from shared link
output = "aksharantar_sampled.zip"

# Download
# gdown.download(f"https://drive.google.com/uc?id={file_id}", output, quiet=False)

# Extract
# !unzip -q aksharantar_sampled.zip -d data/raw/
# print("✅ Dataset downloaded and extracted!")
"""

# CODE CELL 2.2 - Check Dataset Location
# First, let's find where the dataset is
print("🔍 Checking dataset location...\n")

# Check if it exists in Drive
src_path = "/content/drive/MyDrive/aksharantar_sampled"
if os.path.exists(src_path):
    print(f"✅ Found dataset at: {src_path}")
else:
    print(f"❌ Dataset not found at: {src_path}")
    print("\n🔍 Searching in MyDrive root...")
    os.system("ls -la /content/drive/MyDrive/ | grep aksharantar")
    
    # Alternative common locations
    alt_paths = [
        "/content/drive/My Drive/aksharantar_sampled",
        "/content/drive/MyDrive/Colab Notebooks/aksharantar_sampled",
        "/content/aksharantar_sampled"
    ]
    
    for path in alt_paths:
        if os.path.exists(path):
            src_path = path
            print(f"\n✅ Found dataset at: {src_path}")
            break

# CODE CELL 2.3 - Copy Dataset Safely
# Create destination folder
os.makedirs("data/raw", exist_ok=True)

# Copy dataset with proper handling
if not os.path.exists("data/raw/aksharantar_sampled"):
    if os.path.exists(src_path):
        print(f"\n📦 Copying dataset from {src_path}...")
        import shutil
        try:
            shutil.copytree(src_path, "data/raw/aksharantar_sampled")
            print("✅ Dataset copied successfully!")
        except Exception as e:
            print(f"❌ Error copying: {e}")
            print("\n💡 Tip: Make sure the dataset is uploaded to Google Drive")
            print("   Expected structure: MyDrive/aksharantar_sampled/hin/hin_train.csv")
    else:
        print("❌ Source path not found!")
        print("\n📝 Please upload the dataset to Google Drive:")
        print("   1. Download aksharantar_sampled.zip")
        print("   2. Upload to MyDrive root")
        print("   3. Extract the zip file")
        raise FileNotFoundError("Dataset not found in Google Drive")
else:
    print("✅ Dataset already present in data/raw/")

# CODE CELL 2.4 - Verify Dataset Structure
# Verify the folder structure
print("\n📂 Verifying dataset structure:\n")

# Check directory contents
if os.path.exists("data/raw/aksharantar_sampled"):
    print("Contents of data/raw/aksharantar_sampled/:")
    os.system("ls -la data/raw/aksharantar_sampled/")
    
    print("\nContents of hin/ folder:")
    if os.path.exists("data/raw/aksharantar_sampled/hin"):
        os.system("ls -lh data/raw/aksharantar_sampled/hin/")
    else:
        print("❌ hin/ folder not found!")
else:
    print("❌ aksharantar_sampled folder not found!")

# Check file sizes
print("\n📊 File Information:")
for split in ['train', 'valid', 'test']:
    file_path = f"data/raw/aksharantar_sampled/hin/hin_{split}.csv"
    if os.path.exists(file_path):
        size = os.path.getsize(file_path) / 1024  # KB
        num_lines = sum(1 for _ in open(file_path, encoding='utf-8'))
        print(f"   ✅ hin_{split}.csv: {size:.2f} KB ({num_lines:,} lines)")
    else:
        print(f"   ❌ hin_{split}.csv: NOT FOUND")

# CODE CELL 2.5 - Data Loading Function (Robust Version)
def load_data(data_path, split='train'):
    """
    Load Aksharantar dataset from CSV file with robust error handling.
    
    Args:
        data_path: Path to the dataset directory
        split: 'train', 'valid', or 'test'
    
    Returns:
        List of (source, target) tuples where source is Latin, target is Devanagari
    """
    file_path = os.path.join(data_path, f'hin_{split}.csv')
    
    # Check if file exists
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        print(f"   Current directory: {os.getcwd()}")
        print(f"   Looking in: {data_path}")
        
        # Try to find the file
        if os.path.exists('data/raw/aksharantar_sampled'):
            print("\n   Available files:")
            os.system("find data/raw/aksharantar_sampled -name '*.csv'")
        
        raise FileNotFoundError(f"Dataset file not found: {file_path}")
    
    print(f"✅ Loading {split} data from: {file_path}")
    
    # Read CSV without headers
    try:
        df = pd.read_csv(file_path, header=None, names=['devanagari', 'latin'], 
                        encoding='utf-8')
    except Exception as e:
        print(f"❌ Error reading CSV: {e}")
        # Try with different encoding
        df = pd.read_csv(file_path, header=None, names=['devanagari', 'latin'], 
                        encoding='utf-8-sig')
    
    # Clean and reverse the mapping: (latin, devanagari) for our task
    pairs = []
    skipped = 0
    
    for idx, row in df.iterrows():
        if pd.notna(row['latin']) and pd.notna(row['devanagari']):
            latin = str(row['latin']).strip()
            devanagari = str(row['devanagari']).strip()
            
            # Basic validation
            if len(latin) > 0 and len(devanagari) > 0:
                pairs.append((latin, devanagari))
            else:
                skipped += 1
        else:
            skipped += 1
    
    if skipped > 0:
        print(f"   ⚠️  Skipped {skipped} invalid pairs")
    
    return pairs

# CODE CELL 2.6 - Load All Splits
# Define dataset path
DATA_PATH = 'data/raw/aksharantar_sampled/hin'

# Check if directory exists
if not os.path.exists(DATA_PATH):
    print(f"❌ Dataset directory not found: {DATA_PATH}")
    print("\n🔍 Available directories:")
    os.system("ls -R data/raw/")
    raise FileNotFoundError(f"Please ensure dataset is at: {DATA_PATH}")

print("📥 Loading Hindi dataset...\n")

# Load all splits with error handling
try:
    train_pairs = load_data(DATA_PATH, 'train')
    print(f"   Loaded {len(train_pairs):,} training pairs\n")
    
    valid_pairs = load_data(DATA_PATH, 'valid')
    print(f"   Loaded {len(valid_pairs):,} validation pairs\n")
    
    test_pairs = load_data(DATA_PATH, 'test')
    print(f"   Loaded {len(test_pairs):,} test pairs\n")
    
except FileNotFoundError as e:
    print(f"\n❌ Error: {e}")
    print("\n💡 Troubleshooting steps:")
    print("   1. Check if dataset is uploaded to Google Drive")
    print("   2. Verify the folder structure is correct")
    print("   3. Make sure files are named: hin_train.csv, hin_valid.csv, hin_test.csv")
    raise

# Display statistics
print(f"📊 Dataset Statistics:")
print(f"   {'='*50}")
print(f"   Training samples:   {len(train_pairs):,}")
print(f"   Validation samples: {len(valid_pairs):,}")
print(f"   Test samples:       {len(test_pairs):,}")
print(f"   {'='*50}")
print(f"   Total:              {len(train_pairs) + len(valid_pairs) + len(test_pairs):,}")
print(f"   {'='*50}")

# Verify data is loaded correctly
if len(train_pairs) == 0:
    raise ValueError("No training data loaded! Check CSV file format.")

# CODE CELL 2.7 - Display Sample Pairs
print("\n🔍 Sample Transliteration Pairs (Random Selection):\n")
print(f"   {'#':<3} {'Latin (Source)':<25} {'Devanagari (Target)'}")
print(f"   {'-'*60}")

# Ensure we have enough samples
num_samples = min(10, len(train_pairs))
sample_pairs = random.sample(train_pairs, num_samples)

for i, (src, tgt) in enumerate(sample_pairs, 1):
    # Truncate if too long for display
    src_display = src[:24] + "..." if len(src) > 24 else src
    print(f"   {i:<3} {src_display:<25} {tgt}")

print(f"   {'-'*60}")
print(f"   Showing {num_samples} random examples from training set")

# CODE CELL 2.8 - Sequence Length Analysis
def analyze_lengths(pairs, name='Dataset'):
    """Analyze and display sequence length distributions."""
    if len(pairs) == 0:
        print(f"⚠️  {name} is empty!")
        return [], []
    
    src_lengths = [len(src) for src, _ in pairs]
    tgt_lengths = [len(tgt) for _, tgt in pairs]
    
    print(f"\n📏 {name} Length Statistics:")
    print(f"   {'='*60}")
    print(f"   Source (Latin):")
    print(f"      Mean:   {np.mean(src_lengths):.2f} chars")
    print(f"      Median: {np.median(src_lengths):.0f} chars")
    print(f"      Max:    {max(src_lengths)} chars")
    print(f"      Min:    {min(src_lengths)} chars")
    print(f"      Std:    {np.std(src_lengths):.2f}")
    print(f"   Target (Devanagari):")
    print(f"      Mean:   {np.mean(tgt_lengths):.2f} chars")
    print(f"      Median: {np.median(tgt_lengths):.0f} chars")
    print(f"      Max:    {max(tgt_lengths)} chars")
    print(f"      Min:    {min(tgt_lengths)} chars")
    print(f"      Std:    {np.std(tgt_lengths):.2f}")
    print(f"   {'='*60}")
    
    return src_lengths, tgt_lengths

# Analyze training data
src_lens, tgt_lens = analyze_lengths(train_pairs, 'Training Set')

# CODE CELL 2.6 - Visualize Length Distributions
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Latin length distribution
axes[0].hist(src_lens, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
axes[0].axvline(np.mean(src_lens), color='red', linestyle='--', linewidth=2, 
                label=f'Mean: {np.mean(src_lens):.1f}')
axes[0].axvline(np.median(src_lens), color='green', linestyle='--', linewidth=2, 
                label=f'Median: {np.median(src_lens):.0f}')
axes[0].set_xlabel('Sequence Length (characters)', fontsize=11)
axes[0].set_ylabel('Frequency', fontsize=11)
axes[0].set_title('Latin Character Length Distribution', fontsize=13, fontweight='bold')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Devanagari length distribution
axes[1].hist(tgt_lens, bins=30, alpha=0.7, color='lightcoral', edgecolor='black')
axes[1].axvline(np.mean(tgt_lens), color='red', linestyle='--', linewidth=2, 
                label=f'Mean: {np.mean(tgt_lens):.1f}')
axes[1].axvline(np.median(tgt_lens), color='green', linestyle='--', linewidth=2, 
                label=f'Median: {np.median(tgt_lens):.0f}')
axes[1].set_xlabel('Sequence Length (characters)', fontsize=11)
axes[1].set_ylabel('Frequency', fontsize=11)
axes[1].set_title('Devanagari Character Length Distribution', fontsize=13, fontweight='bold')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('length_distribution.png', dpi=300, bbox_inches='tight')
plt.show()

print("✅ Length distribution plot saved as 'length_distribution.png'")

# ═══════════════════════════════════════════════════════════════════════════════

# MARKDOWN CELL 3
"""
---

## 🔤 Step 3: Tokenization and Data Encoding

### 🎯 Vocabulary Construction Strategy

We build **separate vocabularies** for source (Latin) and target (Devanagari):

#### Special Tokens:
- `<PAD>` (index 0): Padding token for batch processing
- `<SOS>` (index 1): Start-of-sequence marker (decoder input)
- `<EOS>` (index 2): End-of-sequence marker (stopping criterion)
- `<UNK>` (index 3): Unknown characters (rare/unseen at test time)

#### Character Coverage:
- **Latin**: a-z, 0-9, common punctuation (normalized to lowercase)
- **Devanagari**: All Unicode characters present in dataset

---

### 🔢 Encoding Pipeline

Each word is converted to a sequence of integer indices:
```python
"ghar" → [5, 8, 1, 18] + [2]  # Adding <EOS>
"घर"   → [1] + [15, 23] + [2]  # Adding <SOS> and <EOS>
```

**Why this encoding?**
- Enables efficient embedding lookups
- Allows GPU-accelerated batch processing
- Maintains sequence boundaries with special tokens

---

### 📦 Dataset Class Design

We create a PyTorch `Dataset` that:
1. Encodes word pairs to index sequences
2. Handles variable-length sequences
3. Filters sequences longer than max_len (50 chars)
4. Supports efficient batching with custom collate function

---
"""

# CODE CELL 3.1 - Vocabulary Class
class Vocabulary:
    """
    Manages character-to-index and index-to-character mappings.
    Handles special tokens and provides encode/decode functionality.
    """
    PAD_TOKEN = '<PAD>'
    SOS_TOKEN = '<SOS>'
    EOS_TOKEN = '<EOS>'
    UNK_TOKEN = '<UNK>'
    
    def __init__(self, name):
        """
        Initialize vocabulary with special tokens.
        
        Args:
            name: Name identifier for this vocabulary (e.g., 'Latin', 'Devanagari')
        """
        self.name = name
        self.char2idx = {
            self.PAD_TOKEN: 0,
            self.SOS_TOKEN: 1,
            self.EOS_TOKEN: 2,
            self.UNK_TOKEN: 3
        }
        self.idx2char = {idx: char for char, idx in self.char2idx.items()}
        self.char_count = Counter()  # Track character frequencies
    
    def add_word(self, word):
        """Add all characters from a word to vocabulary."""
        for char in word:
            self.char_count[char] += 1
            if char not in self.char2idx:
                idx = len(self.char2idx)
                self.char2idx[char] = idx
                self.idx2char[idx] = char
    
    def __len__(self):
        """Return vocabulary size."""
        return len(self.char2idx)
    
    def encode(self, word, add_sos=False, add_eos=False):
        """
        Convert word to list of indices.
        
        Args:
            word: Input string
            add_sos: Whether to prepend <SOS> token
            add_eos: Whether to append <EOS> token
        
        Returns:
            List of integer indices
        """
        indices = []
        
        if add_sos:
            indices.append(self.char2idx[self.SOS_TOKEN])
        
        for char in word:
            # Use <UNK> for unseen characters
            indices.append(self.char2idx.get(char, self.char2idx[self.UNK_TOKEN]))
        
        if add_eos:
            indices.append(self.char2idx[self.EOS_TOKEN])
        
        return indices
    
    def decode(self, indices, remove_special=True):
        """
        Convert list of indices back to word.
        
        Args:
            indices: List of integer indices
            remove_special: Whether to filter out special tokens
        
        Returns:
            Decoded string
        """
        chars = []
        for idx in indices:
            char = self.idx2char.get(idx, self.UNK_TOKEN)
            # Skip special tokens if requested
            if remove_special and char in [self.PAD_TOKEN, self.SOS_TOKEN, 
                                          self.EOS_TOKEN, self.UNK_TOKEN]:
                continue
            chars.append(char)
        return ''.join(chars)

print("✅ Vocabulary class defined!")

# CODE CELL 3.2 - Build Vocabularies
print("🏗️  Building vocabularies from training data...\n")

src_vocab = Vocabulary('Latin')
tgt_vocab = Vocabulary('Devanagari')

# Add all characters from training data only (avoid test set leakage)
for src, tgt in tqdm(train_pairs, desc='Processing training pairs'):
    src_vocab.add_word(src.lower())  # Lowercase for Latin normalization
    tgt_vocab.add_word(tgt)

print(f"\n📚 Vocabulary Statistics:")
print(f"   {'='*60}")
print(f"   Latin vocabulary size:      {len(src_vocab):,} characters")
print(f"   Devanagari vocabulary size: {len(tgt_vocab):,} characters")
print(f"   {'='*60}")

# CODE CELL 3.3 - Display Character Frequencies
print(f"\n🔝 Top 20 Most Frequent Latin Characters:\n")
print(f"   {'Char':<6} {'Frequency':<12} {'Bar'}")
print(f"   {'-'*40}")

for char, count in src_vocab.char_count.most_common(20):
    # Display character (handle special chars)
    display_char = repr(char) if char in [' ', '\t', '\n'] else char
    bar = '█' * (count // 1000)  # Visual bar
    print(f"   {display_char:<6} {count:>10,}   {bar}")

print(f"\n🔝 Top 20 Most Frequent Devanagari Characters:\n")
print(f"   {'Char':<6} {'Frequency':<12} {'Bar'}")
print(f"   {'-'*40}")

for char, count in tgt_vocab.char_count.most_common(20):
    bar = '█' * (count // 1000)
    print(f"   {char:<6} {count:>10,}   {bar}")

# CODE CELL 3.4 - Test Encoding/Decoding
# Test the encode/decode functionality
print("\n🧪 Testing Encode/Decode Functionality:\n")

test_words = [
    ("namaste", "नमस्ते"),
    ("bharat", "भारत"),
    ("hindi", "हिंदी")
]

for src_word, tgt_word in test_words:
    # Encode
    src_encoded = src_vocab.encode(src_word.lower(), add_sos=False, add_eos=True)
    tgt_encoded = tgt_vocab.encode(tgt_word, add_sos=True, add_eos=True)
    
    # Decode back
    src_decoded = src_vocab.decode(src_encoded, remove_special=True)
    tgt_decoded = tgt_vocab.decode(tgt_encoded, remove_special=True)
    
    print(f"   Source: {src_word:<10} → {src_encoded} → {src_decoded}")
    print(f"   Target: {tgt_word:<10} → {tgt_encoded} → {tgt_decoded}")
    print(f"   Match: ✅" if src_word.lower() == src_decoded and tgt_word == tgt_decoded else "   Match: ❌")
    print()

# CODE CELL 3.5 - Dataset Class
class TransliterationDataset(Dataset):
    """
    PyTorch Dataset for transliteration task.
    Handles encoding and filtering of word pairs.
    """
    def __init__(self, pairs, src_vocab, tgt_vocab, max_len=50):
        """
        Args:
            pairs: List of (source, target) string tuples
            src_vocab: Source vocabulary object
            tgt_vocab: Target vocabulary object
            max_len: Maximum sequence length (for filtering)
        """
        # Filter pairs by length and normalize source to lowercase
        self.pairs = [
            (s.lower(), t) for s, t in pairs 
            if len(s) <= max_len and len(t) <= max_len
        ]
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        
        print(f"   Filtered {len(pairs) - len(self.pairs)} pairs exceeding max_len={max_len}")
        print(f"   Remaining pairs: {len(self.pairs)}")
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        """
        Get a single encoded pair.
        
        Returns:
            src_tensor: Source indices with <EOS>
            tgt_tensor: Target indices with <SOS> and <EOS>
        """
        src, tgt = self.pairs[idx]
        
        # Encode source: no <SOS>, with <EOS>
        src_indices = self.src_vocab.encode(src, add_sos=False, add_eos=True)
        
        # Encode target: with <SOS> and <EOS>
        tgt_indices = self.tgt_vocab.encode(tgt, add_sos=True, add_eos=True)
        
        return (torch.LongTensor(src_indices), 
                torch.LongTensor(tgt_indices))

print("✅ TransliterationDataset class defined!")

# CODE CELL 3.6 - Custom Collate Function
def collate_fn(batch):
    """
    Custom collate function to pad sequences in a batch to same length.
    
    Args:
        batch: List of (src_tensor, tgt_tensor) tuples
    
    Returns:
        src_batch: (batch_size, max_src_len) - padded source sequences
        src_lengths: (batch_size,) - actual source lengths
        tgt_batch: (batch_size, max_tgt_len) - padded target sequences
        tgt_lengths: (batch_size,) - actual target lengths
    """
    src_batch, tgt_batch = zip(*batch)
    
    # Get lengths before padding (needed for pack_padded_sequence)
    src_lengths = torch.LongTensor([len(s) for s in src_batch])
    tgt_lengths = torch.LongTensor([len(t) for t in tgt_batch])
    
    # Pad sequences to max length in batch (padding_value=0 for <PAD>)
    src_batch = pad_sequence(src_batch, batch_first=True, padding_value=0)
    tgt_batch = pad_sequence(tgt_batch, batch_first=True, padding_value=0)
    
    return src_batch, src_lengths, tgt_batch, tgt_lengths

print("✅ Custom collate function defined!")

# CODE CELL 3.7 - Create Datasets
MAX_SEQ_LEN = 50  # Filter sequences longer than this

print(f"\n📦 Creating PyTorch Datasets (max_len={MAX_SEQ_LEN})...\n")

print("Training set:")
train_dataset = TransliterationDataset(train_pairs, src_vocab, tgt_vocab, MAX_SEQ_LEN)

print("\nValidation set:")
valid_dataset = TransliterationDataset(valid_pairs, src_vocab, tgt_vocab, MAX_SEQ_LEN)

print("\nTest set:")
test_dataset = TransliterationDataset(test_pairs, src_vocab, tgt_vocab, MAX_SEQ_LEN)

# CODE CELL 3.8 - Create DataLoaders
BATCH_SIZE = 128

print(f"\n📊 Creating DataLoaders (batch_size={BATCH_SIZE})...\n")

train_loader = DataLoader(
    train_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=True,  # Shuffle training data
    collate_fn=collate_fn, 
    num_workers=2,  # Parallel data loading
    pin_memory=True  # Faster GPU transfer
)

valid_loader = DataLoader(
    valid_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=False,  # No shuffle for validation
    collate_fn=collate_fn, 
    num_workers=2, 
    pin_memory=True
)

test_loader = DataLoader(
    test_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=False,  # No shuffle for test
    collate_fn=collate_fn, 
    num_workers=2, 
    pin_memory=True
)

print(f"✅ DataLoaders created successfully!")
print(f"\n   Training batches:   {len(train_loader)}")
print(f"   Validation batches: {len(valid_loader)}")
print(f"   Test batches:       {len(test_loader)}")

# CODE CELL 3.9 - Test DataLoader
# Test a single batch
print(f"\n🧪 Testing DataLoader (first batch):\n")

src_batch, src_lens, tgt_batch, tgt_lens = next(iter(train_loader))

print(f"   Source batch shape: {src_batch.shape}")
print(f"   Source lengths shape: {src_lens.shape}")
print(f"   Target batch shape: {tgt_batch.shape}")
print(f"   Target lengths shape: {tgt_lens.shape}")

# Display first example
print(f"\n   First example:")
print(f"      Source indices: {src_batch[0].tolist()}")
print(f"      Source decoded: '{src_vocab.decode(src_batch[0].tolist())}'")
print(f"      Target indices: {tgt_batch[0].tolist()}")
print(f"      Target decoded: '{tgt_vocab.decode(tgt_batch[0].tolist())}'")

print(f"\n✅ Data encoding pipeline complete!")

# ═══════════════════════════════════════════════════════════════════════════════

# MARKDOWN CELL 4
"""
---

## 🏗️ Step 4: Model Architecture (Encoder-Decoder with Attention)

### 🎯 Architecture Overview

Our seq2seq model consists of **three interconnected components**:

---

#### 1️⃣ **Bidirectional LSTM Encoder**

**Purpose**: Encode input Latin character sequence into fixed-dimensional representations

**Architecture**:
```
Input: [c₁, c₂, ..., cₙ] (Latin characters)
       ↓
  Embedding Layer (E-dimensional)
       ↓
  Bidirectional LSTM (2 × H hidden units)
       ↓
Output: [h₁, h₂, ..., hₙ] (contextual representations)
        [h_final, c_final] (final states for decoder initialization)
```

**Why Bidirectional?**
- Captures context from both left→right and right→left
- Better understanding of character relationships
- Richer representations for attention mechanism

---

#### 2️⃣ **Bahdanau Attention Mechanism**

**Purpose**: Dynamically focus on relevant encoder states during decoding

**Mathematical Formulation**:
```
score(hₜ, h̄ₛ) = vᵀ · tanh(W₁·hₜ + W₂·h̄ₛ)
αₜₛ = exp(score(hₜ, h̄ₛ