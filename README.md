# We are Closed Until Further Notice

![IMG_66FBF6EF-DC37-461F-BCA0-721203DFA9E1](https://github.com/user-attachments/assets/6baac54e-e555-4353-9437-9cbeea00b2db)

## Overview
This project aims to revitalize a long-abandoned movie theatre marquee using exact anagrams of the fixed phrase: **"WE ARE CLOSED UNTIL FURTHER NOTICE STAY HEALTHY"**. The goal is to generate grammatically valid and emotionally resonant alternatives using only the original letters, then curate the most meaningful messages to display in a public space.

The process balances automation (for anagram generation and validation) with human-AI collaboration (for meaning-making and curation).

## Project Flow

1. **Input Phrase**: User defines the fixed phrase.
2. **Anagram Generation**: Python CLI (`anagram_cli.py`) takes the phrase, filters a dictionary based on rules, and returns valid anagrammatic sentences.
3. **Validation**: Each result is guaranteed to use all original letters with no additions or omissions.
4. **Curation**: A second CLI (`analyze_anagrams.py`) sends the generated list to OpenAI's API with a prompt and criteria for ranking.
5. **Selection**: The user selects top variants for real-world deployment.

## Key Features
- Dictionary filtering by max word length, prefixes, and suffixes
- Deterministic letter usage (no letter replacement allowed)
- Exportable results (CSV or TXT)
- OpenAI-powered interpretation with user-defined ranking criteria

## Usage
See `requirements.txt` to install dependencies.
