# FIDA Normalizer

A text normalization package for TTS (Text-to-Speech) preprocessing with multi-language support.

## Features

- **Multi-language Support**: Supports Italian, English, French, German, Spanish, Portuguese, Dutch, and Swedish
- **Text Normalization**: Converts numbers, dates, times, emails, URLs, and special characters to spoken forms
- **Phonemization Support**: Optional IPA phoneme generation using Espeak
- **TTS Integration**: Designed to work with NeMo and other TTS systems
- **Currency & Units**: Handles currency symbols and measurement units

## Installation

### From PyPI (Recommended)

```bash
pip install fida-normalizer
```

### From Source

```bash
git clone <repository-url>
cd fida_tts
pip install -e .
```

### For Development

```bash
pip install -e ".[dev]"
```

## Usage

### Basic Usage

```python
from Normalizer import Normalizer

# Create a normalizer instance (Italian language by default)
normalizer = Normalizer(lang='it')

# Normalize text
result = normalizer.normalize("Il prezzo è 50,50 euro")
print(result)  # Output: il prezzo è cinquanta virgola cinquanta euro
```

### With Different Languages

```python
from Normalizer import Normalizer

# English normalizer
en_normalizer = Normalizer(lang='en')
result = en_normalizer.normalize("The price is $100.50")
print(result)  # Output: the price is dollar one hundred point fifty
```

### With Phonemization

```python
from Normalizer import Normalizer

# Enable phonemization (requires espeak/espeak-ng installed)
normalizer = Normalizer(lang='it', phonemize=True)
result = normalizer.normalize("Ciao mondo")
print(result)  # Output: IPA phonemes
```

### Configuration Options

```python
from Normalizer import Normalizer

normalizer = Normalizer(
    lang='it',           # Language code: 'it', 'en', 'fr', 'de', 'es', 'pt', 'nl', 'sv'
    tts_mode=True,       # TTS mode: keeps punctuation suitable for TTS
    to_lower=True,       # Convert output to lowercase
    phonemize=False      # Enable/disable phonemization
)
```

## Supported Languages

| Language | Code |
|----------|------|
| Italian | `it` |
| English | `en` |
| French | `fr` |
| German | `de` |
| Spanish | `es` |
| Portuguese | `pt` |
| Dutch | `nl` |
| Swedish | `sv` |

## Supported Transformations

- **Numbers**: Converts digits to spoken words (e.g., "123" → "one hundred twenty-three")
- **Decimals**: Handles decimal separators based on locale
- **Percentages**: Converts "%" to spoken form
- **Currency**: Handles $, €, £, ¥ symbols
- **Dates**: Converts date formats to spoken form
- **Times**: Converts time formats (HH:MM, HH:MM:SS)
- **Emails**: Spells out email addresses
- **URLs/Domains**: Spells out web addresses
- **Units**: Converts measurement units (m, kg, km/h, etc.)
- **Special Characters**: Handles @, &, -, _, /, etc.

## Requirements

- Python >= 3.8
- phonemizer >= 3.0.0 (optional, for phonemization)
- espeak or espeak-ng (system dependency for phonemization)

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
