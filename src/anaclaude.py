import argparse
import os
import re
import time
import json
import random
import getpass
from collections import Counter, defaultdict
import requests
import numpy as np
from typing import List, Dict, Tuple, Set, Optional

def normalize_text(text: str) -> str:
    """Remove non-alphabetic characters and convert to lowercase"""
    return re.sub(r'[^a-z]', '', text.lower())

def can_form_word(word: str, available_letters: Counter) -> bool:
    """Check if a word can be formed from available letters"""
    word_counter = Counter(normalize_text(word))
    return all(word_counter[char] <= available_letters[char] for char in word_counter)

def update_available_letters(available_letters: Counter, word: str) -> Counter:
    """Update available letters after using a word"""
    word_counter = Counter(normalize_text(word))
    result = available_letters.copy()
    for char, count in word_counter.items():
        result[char] -= count
    return result

def calculate_fitness(phrase: str, target_phrase) -> Tuple[float, int, List[str], List[str]]:
    """
    Calculate fitness score for a phrase compared to target phrase.
    
    Returns:
    - fitness: Score indicating how close the phrase is to a perfect anagram
    - unused_count: Number of unused letters
    - unused_letters: List of unused letters
    - excess_letters: List of excess letters used
    """
    phrase_counter = Counter(normalize_text(phrase))
    # Handle if target_phrase is already a Counter object
    if isinstance(target_phrase, Counter):
        target_counter = target_phrase
    else:
        target_counter = Counter(normalize_text(target_phrase))
    
    # Track unused and excess letters
    unused_letters = []
    excess_letters = []
    
    # Check if any letter is used too many times
    has_excess = False
    for char, count in phrase_counter.items():
        if count > target_counter.get(char, 0):
            has_excess = True
            excess_letters.extend([char] * (count - target_counter.get(char, 0)))
    
    # If phrase uses more letters than available, it's an immediate fail
    if has_excess:
        return 0.0, 0, [], excess_letters
    
    # Count unused letters
    unused_count = 0
    for char, count in target_counter.items():
        diff = count - phrase_counter.get(char, 0)
        if diff > 0:
            unused_count += diff
            unused_letters.extend([char] * diff)
    
    # Calculate fitness score: higher is better
    # Perfect score is 1.0 (when unused_count is 0)
    if unused_count == 0:
        fitness = 1.0
    else:
        # Penalty increases quadratically with unused letters
        fitness = 1.0 / (1.0 + unused_count ** 2)
    
    return fitness, unused_count, unused_letters, excess_letters

def verify_anagram(phrase: str, target_phrase: str) -> bool:
    """Verify that a phrase is a valid anagram of the target"""
    phrase_counter = Counter(normalize_text(phrase))
    target_counter = Counter(normalize_text(target_phrase))
    return phrase_counter == target_counter

def load_dictionary(dict_file: str, min_length: int = 1, include_common: bool = True) -> List[str]:
    """Load dictionary from file, filtering by minimum length.
    Optionally include a list of common English words."""
    words = []
    with open(dict_file, 'r', encoding='utf-8') as f:
        words = [line.strip() for line in f if len(line.strip()) >= min_length]
    
    # Include common short words even if they're not in the dictionary
    if include_common and min_length <= 3:
        common_words = ["a", "an", "as", "at", "be", "by", "do", "for", "he", "her", "his", "how", 
                        "if", "in", "is", "it", "my", "no", "not", "of", "on", "or", "our", "so", 
                        "the", "to", "up", "us", "we", "see", "say", "use", "all", "and", 
                        "are", "but", "can", "day", "did", "end", "far", "few", "fun", "get", 
                        "has", "had", "her", "him", "his", "how", "let", "may", "new", "now", 
                        "off", "old", "one", "out", "own", "run", "sat", "say", "see", "set", 
                        "she", "two", "use", "was", "way", "who", "why", "yes", "yet", "you"]
        for word in common_words:
            if word not in words and len(word) >= min_length:
                words.append(word)
                
    return list(set(words))  # Remove duplicates

def filter_constructible_words(dictionary: List[str], target_letters: Counter) -> List[str]:
    """Filter dictionary to only include words that can be constructed from target letters"""
    return [word for word in dictionary if can_form_word(word, target_letters)]

def group_by_first_letter(words: List[str]) -> Dict[str, List[str]]:
    """Group words by their first letter for more efficient lookup"""
    groups = defaultdict(list)
    for word in words:
        if word:
            groups[word[0]].append(word)
    return groups

def generate_two_word_combinations(constructible_words: List[str], target_letters: Counter) -> List[Tuple[str, float, int]]:
    """
    Generate anagrams using two words.
    Returns list of (phrase, fitness_score, unused_letters_count) tuples.
    """
    word_groups = group_by_first_letter(constructible_words)
    results = []
    
    # Sort words by length (descending) for better combinations
    sorted_words = sorted(constructible_words, key=len, reverse=True)
    
    print("Generating two-word combinations...")
    count = 0
    total = min(len(sorted_words), 1000)  # Limit to prevent excessive combinations
    
    # Track near-perfect anagrams
    near_perfect = []
    
    for i, word1 in enumerate(sorted_words[:total]):
        if i % 100 == 0:
            print(f"  Processed {i}/{total} words...")
        
        remaining_letters = update_available_letters(target_letters, word1)
        
        # Try to find a second word
        for word2 in constructible_words:
            if word2 != word1:
                phrase = f"{word1} {word2}"
                
                # Calculate fitness and check if it's valid
                fitness, unused_count, unused_letters, excess_letters = calculate_fitness(phrase, target_letters)
                
                # If we're using excess letters, skip
                if excess_letters:
                    continue
                
                # Perfect match
                if unused_count == 0:
                    results.append((phrase, fitness, unused_count))
                    count += 1
                    if count % 10 == 0:
                        print(f"  Found {count} perfect two-word combinations")
                
                # Near perfect (less than 5 unused letters)
                elif unused_count <= 5 and fitness > 0.04:  # Threshold can be adjusted
                    near_perfect.append((phrase, fitness, unused_count))
    
    # Add some of the best near-perfect results
    near_perfect.sort(key=lambda x: x[1], reverse=True)
    results.extend(near_perfect[:min(20, len(near_perfect))])
    
    print(f"Found {len(results)} two-word combinations ({count} perfect, {len(results)-count} near-perfect)")
    return results

def get_api_key(args):
    """Get API key from various sources, prompting if necessary"""
    # Check command line argument first
    if args.api_key:
        return args.api_key
    
    # Check environment variable
    if os.environ.get("CLAUDE_API_KEY"):
        return os.environ.get("CLAUDE_API_KEY")
    
    # Check for key file
    key_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.api_key')
    try:
        with open(key_file, 'r') as f:
            return f.read().strip()
    except (FileNotFoundError, IOError):
        pass
    
    # Prompt user for key
    print("Claude API key not found. You can get one from https://console.anthropic.com/")
    api_key = getpass.getpass("Enter your Claude API key: ")
    
    # Optionally save for future use
    save = input("Save key for future use? (y/n): ").lower()
    if save.startswith('y'):
        try:
            with open(key_file, 'w') as f:
                f.write(api_key)
            os.chmod(key_file, 0o600)  # Restrict file permissions
            print(f"API key saved to {key_file}")
        except IOError as e:
            print(f"Warning: Could not save API key: {e}")
    
    return api_key

def call_llm_api(prompt: str, api_key: str, model: str = "claude-3-opus-20240229", 
                max_retries: int = 3, retry_delay: int = 5) -> str:
    """Call the Claude API with a prompt and return the response with retries"""
    headers = {
        "x-api-key": api_key,
        "content-type": "application/json",
        "anthropic-version": "2023-06-01"
    }
    
    payload = {
        "model": model,
        "max_tokens": 1000,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7,
    }
    
    for attempt in range(max_retries):
        try:
            response = requests.post(
                "https://api.anthropic.com/v1/messages",
                headers=headers,
                json=payload,
                timeout=60
            )
            
            if response.status_code == 200:
                return response.json()["content"][0]["text"]
            elif response.status_code == 429:
                wait_time = retry_delay * (2 ** attempt)
                print(f"Rate limited. Waiting {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                print(f"Error: {response.status_code}")
                print(response.text)
                break
                
        except requests.exceptions.RequestException as e:
            print(f"API request failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay * (2 ** attempt))
            else:
                print("Max retries reached.")
    
    return ""

def call_llm_mock_api(prompt: str, api_key: str, model: str = "claude-3-opus-20240229", 
                max_retries: int = 3, retry_delay: int = 5) -> str:
    """Call the Claude API with a prompt and return the response with retries"""
    # For debugging, return a mock response instead of making a real API call
    print("Mock API call made - using debug mode")
    
    # Different mock responses based on the type of prompt
    if "seed words" in prompt:
        return '["theater", "show", "movies", "time", "night", "watch", "fun", "we", "the", "a"]'
    elif "suggest completions" in prompt or "complete phrases" in prompt:
        return '["Example completion 1", "Example completion 2", "Example completion 3"]'
    elif "evaluate the following anagram phrases" in prompt:
        return '{"Example completion 1": 8.5, "Example completion 2": 7.2, "Example completion 3": 6.8}'
    else:
        return '["Default mock response"]'

def get_semantic_seed_words(target_phrase: str, constructible_words: List[str], api_key: str, n: int = 20) -> List[str]:
    """Use LLM to identify semantically relevant seed words from constructible words"""
    # Format the list of constructible words for the prompt
    # Include more words for better coverage
    word_sample = random.sample(constructible_words, min(500, len(constructible_words)))
    
    # Group words by length for better representation
    short_words = [w for w in word_sample if len(w) <= 3]
    medium_words = [w for w in word_sample if 4 <= len(w) <= 6]
    long_words = [w for w in word_sample if len(w) > 6]
    
    # Create a balanced sample
    balanced_sample = []
    balanced_sample.extend(random.sample(short_words, min(100, len(short_words))))
    balanced_sample.extend(random.sample(medium_words, min(250, len(medium_words))))
    balanced_sample.extend(random.sample(long_words, min(150, len(long_words))))
    
    words_str = ", ".join(balanced_sample)
    
    prompt = f"""I need to create anagrams for a movie theater marquee that currently says:
"{target_phrase}"

I can only use the existing letters on the marquee (exactly once each).

From the list of words below, identify {n} words that would be good semantic anchors or starting points for creating meaningful phrases that could work on a theater marquee or civic sign. Choose words that capture key concepts or themes related to entertainment, community, or uplifting messages.

Include a diverse mix of:
1. Longer content words (nouns, verbs, adjectives)
2. Shorter utility words (articles, prepositions) that help form complete phrases
3. Words specifically related to theaters, films, or entertainment if available

Available words: {words_str}

Please return only a JSON array of the {n} words, nothing else. For example: ["word1", "word2", "word3", ...]"""

    response = call_llm_api(prompt, api_key)
    
    # Extract the JSON array from the response
    try:
        # Look for anything that looks like a JSON array
        json_match = re.search(r'\[\s*"[^"]*"(?:\s*,\s*"[^"]*")*\s*\]', response)
        if json_match:
            seed_words = json.loads(json_match.group(0))
        else:
            # Fall back to parsing the full response
            seed_words = json.loads(response)
        return seed_words
    except json.JSONDecodeError:
        print("Error parsing response as JSON. Raw response:")
        print(response)
        # Fall back to simple parsing
        words = re.findall(r'"([^"]+)"', response)
        return words[:n]

def llm_guided_completion(seed_word: str, remaining_letters: Counter, target_phrase: str, api_key: str, n: int = 5, 
                         allow_near_perfect: bool = True) -> List[Tuple[str, float, int, List[str]]]:
    """
    Use LLM to suggest completions for a seed word given remaining letters.
    Returns list of (phrase, fitness_score, unused_letters_count, unused_letters) tuples.
    """
    # Convert remaining letters to a string representation
    letters_str = ""
    for char, count in sorted(remaining_letters.items()):
        letters_str += char * count
    
    # If the remaining letters are few, include all possible permutations for reference
    remaining_letter_count = sum(remaining_letters.values())
    all_permutations_str = ""
    if remaining_letter_count <= 7:
        import itertools
        letter_list = []
        for char, count in remaining_letters.items():
            letter_list.extend([char] * count)
        # Get a sample of permutations
        perms = list(itertools.permutations(letter_list))
        sample_perms = random.sample(perms, min(10, len(perms)))
        all_permutations_str = "\nSome permutations of these letters: " + ", ".join([''.join(p) for p in sample_perms])
    
    extra_instructions = ""
    if allow_near_perfect:
        extra_instructions = """
You can also suggest phrases that use ALMOST all remaining letters, leaving out up to 4 letters if it helps create a more meaningful phrase. If you do this, indicate which letters are unused in parentheses at the end of your suggestion.
For example: "phrase that uses most letters (unused: xyz)"
"""
    
    prompt = f"""I'm creating an anagram for a movie theater marquee that currently says:
"{target_phrase}"

I've already chosen the word "{seed_word}" as a starting point. I need to use exactly these remaining letters to complete the anagram: "{letters_str}"{all_permutations_str}

Please suggest {n} complete, grammatical phrases that:
1. Use the word "{seed_word}"
2. Use EXACTLY all the remaining letters - no more, no less
3. Would be appropriate for a theater marquee or public sign
4. Are clear, meaningful, and ideally uplifting or entertaining{extra_instructions}

Be creative! Try different word arrangements and punctuation. Remember that a theater marquee often announces shows, events, or general messages to the community.

Return your answer as a JSON array of complete phrases, nothing else. For example: ["full phrase 1", "full phrase 2", "full phrase 3"]"""

    response = call_llm_api(prompt, api_key)
    
    # Extract completion candidates
    try:
        # Look for anything that looks like a JSON array
        json_match = re.search(r'\[\s*"[^"]*"(?:\s*,\s*"[^"]*")*\s*\]', response)
        if json_match:
            completions = json.loads(json_match.group(0))
        else:
            # Fall back to parsing the full response
            completions = json.loads(response)
    except json.JSONDecodeError:
        print("Error parsing response as JSON. Raw response:")
        print(response)
        # Fall back to simple parsing
        completions = re.findall(r'"([^"]+)"', response)[:n]
    
    # Process completions and calculate fitness
    result_completions = []
    target_normalized = normalize_text(target_phrase)
    
    for completion in completions:
        # Extract unused letters if specified
        unused_info = re.search(r'\(unused:?\s*([a-z\s,]+)\)', completion.lower())
        if unused_info:
            # Clean up the phrase by removing the unused letter information
            phrase = re.sub(r'\s*\(unused:?\s*[a-z\s,]+\)', '', completion)
        else:
            phrase = completion
            
        # Calculate fitness
        fitness, unused_count, unused_letters, excess_letters = calculate_fitness(phrase, target_normalized)
        
        # Skip if using excess letters
        if excess_letters:
            continue
            
        result_completions.append((phrase, fitness, unused_count, unused_letters))
        
    return result_completions

def multi_seed_completion(seed_words: List[str], remaining_letters: Counter, target_phrase: str, api_key: str, n: int = 3,
                        allow_near_perfect: bool = True) -> List[Tuple[str, float, int, List[str]]]:
    """Use LLM to suggest completions using multiple seed words"""
    # Convert remaining letters to a string representation
    letters_str = ""
    for char, count in sorted(remaining_letters.items()):
        letters_str += char * count
    
    seeds_str = ", ".join(seed_words)
    
    extra_instructions = ""
    if allow_near_perfect:
        extra_instructions = """
You can also suggest phrases that use ALMOST all remaining letters, leaving out up to 4 letters if it helps create a more meaningful phrase. If you do this, indicate which letters are unused in parentheses at the end of your suggestion.
For example: "phrase that uses most letters (unused: xyz)"
"""
    
    prompt = f"""I'm creating an anagram for a movie theater marquee that currently says:
"{target_phrase}"

I've already chosen these words as starting points: "{seeds_str}"
I need to use exactly these remaining letters to complete the anagram: "{letters_str}"

Please suggest {n} complete, grammatical phrases that:
1. Use ALL of these words: {seeds_str}
2. Use EXACTLY all the remaining letters - no more, no less
3. Would be appropriate for a theater marquee or public sign
4. Are clear, meaningful, and ideally uplifting or entertaining{extra_instructions}

Be creative with word arrangement and punctuation!

Return your answer as a JSON array of complete phrases, nothing else. For example: ["full phrase 1", "full phrase 2", "full phrase 3"]"""

    response = call_llm_api(prompt, api_key)
    
    # Extract completion candidates
    try:
        json_match = re.search(r'\[\s*"[^"]*"(?:\s*,\s*"[^"]*")*\s*\]', response)
        if json_match:
            completions = json.loads(json_match.group(0))
        else:
            completions = json.loads(response)
    except json.JSONDecodeError:
        print("Error parsing response as JSON. Raw response:")
        print(response)
        completions = re.findall(r'"([^"]+)"', response)[:n]
    
    # Process completions and calculate fitness
    result_completions = []
    target_normalized = normalize_text(target_phrase)
    
    for completion in completions:
        # Extract unused letters if specified
        unused_info = re.search(r'\(unused:?\s*([a-z\s,]+)\)', completion.lower())
        if unused_info:
            # Clean up the phrase by removing the unused letter information
            phrase = re.sub(r'\s*\(unused:?\s*[a-z\s,]+\)', '', completion)
        else:
            phrase = completion
            
        # Calculate fitness
        fitness, unused_count, unused_letters, excess_letters = calculate_fitness(phrase, target_normalized)
        
        # Skip if using excess letters
        if excess_letters:
            continue
            
        result_completions.append((phrase, fitness, unused_count, unused_letters))
    
    return result_completions

def try_seed_pairs(constructible_words: List[str], target_letters: Counter, api_key: str, 
                 num_pairs: int = 10, completions_per_pair: int = 3,
                 allow_near_perfect: bool = True, target_phrase: str = None) -> List[Tuple[str, float, int, List[str]]]:
    """Try pairs of seed words to generate anagrams"""
    # Select words that use a good portion of the letters
    longer_words = [w for w in constructible_words if len(w) >= 5]
    if not longer_words:
        longer_words = constructible_words
        
    # Sort by length for better pairing
    longer_words.sort(key=len, reverse=True)
    
    candidates = []
    pair_count = 0
    
    # Try pairs of words
    for i, word1 in enumerate(longer_words):
        if pair_count >= num_pairs:
            break
            
        word1_letters = Counter(normalize_text(word1))
        remaining1 = target_letters - word1_letters
        
        # If word1 uses too many letters, skip
        if any(count < 0 for count in remaining1.values()):
            continue
            
        for word2 in longer_words[i+1:]:
            if pair_count >= num_pairs:
                break
                
            # Check if word2 can be formed from remaining letters
            word2_letters = Counter(normalize_text(word2))
            remaining2 = remaining1.copy()
            for char, count in word2_letters.items():
                remaining2[char] -= count
                
            # If any letter count is negative, word2 can't be formed
            if any(count < 0 for count in remaining2.values()):
                continue
                
            # We have a valid pair of seed words
            pair_count += 1
            print(f"\nTrying seed pair {pair_count}/{num_pairs}: '{word1}' and '{word2}'")
            
            # Generate completions with this pair
            completions = multi_seed_completion([word1, word2], remaining2, 
                                              target_phrase, api_key, completions_per_pair,
                                              allow_near_perfect)
            
            # Add all completions to candidates
            for completion in completions:
                phrase, fitness, unused_count, unused_letters = completion
                print(f"  Checking: {phrase}")
                print(f"  Fitness: {fitness:.4f}, Unused: {unused_count} letters: {', '.join(unused_letters)}")
                candidates.append(completion)
    
    return candidates

def score_anagrams(candidates: List[Tuple[str, float, int, List[str]]], target_phrase: str, api_key: str) -> List[Tuple[str, float, float, int, List[str]]]:
    """Use LLM to score candidate anagrams for meaningfulness and appropriateness"""
    if not candidates:
        return []
    
    # Batch candidates to avoid too long prompts
    batch_size = 5
    results = []
    
    # Extract just the phrases for scoring
    phrases = [c[0] for c in candidates]
    
    for i in range(0, len(phrases), batch_size):
        batch = phrases[i:i+batch_size]
        candidates_str = "\n".join([f"{j+1}. {candidate}" for j, candidate in enumerate(batch)])
        
        prompt = f"""Please evaluate the following anagram phrases for a movie theater marquee that currently says:
"{target_phrase}"

{candidates_str}

Rate each anagram on a scale of 1-10 based on:
- Grammatical correctness (is it a properly formed sentence or phrase?)
- Clarity (is the meaning clear and easily understood?)
- Appropriateness (would it work well on a public theater marquee?)
- Emotional impact (is it engaging, uplifting, or thought-provoking?)

For each anagram, provide only a single overall score from 1-10 and nothing else.
Return your answer as a JSON object with the anagram as the key and the score as the value. For example:
{{"anagram 1": 7.5, "anagram 2": 8.2, "anagram 3": 6.3}}"""

        response = call_llm_api(prompt, api_key)
        
        try:
            # Extract anything that looks like a JSON object
            json_match = re.search(r'\{(?:[^{}]|"(?:\\.|[^"\\])*")*\}', response)
            if json_match:
                scores = json.loads(json_match.group(0))
            else:
                scores = json.loads(response)
            
            # Add scores to results
            for idx, phrase in enumerate(batch):
                # Find the original candidate data
                original_idx = phrases.index(phrase)
                _, fitness, unused_count, unused_letters = candidates[original_idx]
                
                if phrase in scores:
                    semantic_score = float(scores[phrase])
                else:
                    # If not found by exact match, try to find by number
                    key = f"anagram {idx+1}"
                    if key in scores:
                        semantic_score = float(scores[key])
                    else:
                        semantic_score = 5.0  # Default score
                
                # Add to results with both fitness and semantic scores
                results.append((phrase, fitness, semantic_score, unused_count, unused_letters))
        except json.JSONDecodeError:
            print("Error parsing scores as JSON. Raw response:")
            print(response)
            # Fallback: assign neutral scores
            for phrase in batch:
                # Find the original candidate data
                original_idx = phrases.index(phrase)
                _, fitness, unused_count, unused_letters = candidates[original_idx]
                results.append((phrase, fitness, 5.0, unused_count, unused_letters))
    
    return results

def calculate_combined_score(fitness: float, semantic_score: float, unused_count: int) -> float:
    """Calculate a combined score that balances fitness and semantic qualities"""
    # Convert semantic score to 0-1 range
    semantic_normalized = semantic_score / 10.0
    
    # Perfect anagrams get a boost
    perfect_bonus = 1.0 if unused_count == 0 else 0.0
    
    # Weight factors - adjust these to change the importance of each component
    fitness_weight = 0.5
    semantic_weight = 0.5
    
    # Calculate combined score
    combined_score = (fitness * fitness_weight) + (semantic_normalized * semantic_weight) + perfect_bonus
    
    # Normalize to 0-10 range for final output
    return combined_score * 3.33  # Scale to approximately 0-10


def generate_anagrams(target_phrase: str, dictionary_path: str, api_key: str, 
                      num_seeds: int = 20, num_completions: int = 5, 
                      min_length: int = 1, 
                      try_two_word: bool = True,
                      try_pairs: bool = True,
                      num_pairs: int = 10,
                      allow_near_perfect: bool = True,
                      near_perfect_threshold: int = 4) -> List[Tuple]:
    """Generate anagrams using LLM-guided approach with multiple strategies"""
    target_letters = Counter(normalize_text(target_phrase))
    
    # Load and filter dictionary
    dictionary = load_dictionary(dictionary_path, min_length)
    constructible_words = filter_constructible_words(dictionary, target_letters)
    print(f"Found {len(constructible_words)} constructible words from dictionary.")
    
    # Try algorithm-based approach for two-word anagrams first
    all_candidates = []
    if try_two_word:
        two_word_anagrams = generate_two_word_combinations(constructible_words, target_letters)
        # Convert to the format needed for final scoring
        for phrase, fitness, unused_count in two_word_anagrams:
            _, _, unused_letters, _ = calculate_fitness(phrase, normalize_text(target_phrase))
            all_candidates.append((phrase, fitness, unused_count, unused_letters))
        print(f"Found {len(two_word_anagrams)} algorithmic two-word anagrams.")
    
    # Get semantic seed words
    print("\nGetting semantic seed words...")
    seed_words = get_semantic_seed_words(target_phrase, constructible_words, api_key, num_seeds)
    print(f"Selected seed words: {seed_words}")
    
    # Generate completions for each seed word
    for seed in seed_words:
        print(f"\nWorking with seed word: {seed}")
        # Check if seed word can be formed from target letters
        if not can_form_word(seed, target_letters):
            print(f"  Warning: Seed word '{seed}' cannot be formed from target letters. Skipping.")
            continue
        
        # Calculate remaining letters
        remaining_letters = update_available_letters(target_letters, seed)
        
        # Get LLM-guided completions
        print(f"  Getting completions with remaining letters...")
        completions = llm_guided_completion(seed, remaining_letters, target_phrase, api_key, 
                                         num_completions, allow_near_perfect)
        
        # Process and add completions
        for completion in completions:
            phrase, fitness, unused_count, unused_letters = completion
            
            # Only include if it's perfect or meets the near-perfect threshold
            if unused_count == 0 or (allow_near_perfect and unused_count <= near_perfect_threshold):
                print(f"  Found: {phrase}")
                print(f"  Fitness: {fitness:.4f}, Unused: {unused_count} letters: {', '.join(unused_letters)}")
                all_candidates.append(completion)
    
    # Try pairs of seed words
    if try_pairs:
        print("\nTrying pairs of seed words...")
        pair_candidates = try_seed_pairs(constructible_words, target_letters, api_key, 
                                      num_pairs=num_pairs, completions_per_pair=3,
                                      allow_near_perfect=allow_near_perfect, target_phrase=target_phrase)
        all_candidates.extend(pair_candidates)
    
    # Score and rank candidates
    if all_candidates:
        print("\nScoring candidates...")
        scored_candidates = score_anagrams(all_candidates, target_phrase, api_key)
        
        # Calculate combined scores
        final_candidates = []
        for phrase, fitness, semantic_score, unused_count, unused_letters in scored_candidates:
            combined_score = calculate_combined_score(fitness, semantic_score, unused_count)
            final_candidates.append((phrase, fitness, semantic_score, combined_score, unused_count, unused_letters))
        
        # Sort by combined score (descending)
        return sorted(final_candidates, key=lambda x: x[3], reverse=True)
    
    return []

def display_letter_counts(target_phrase: str):
    """Display letter counts for the target phrase"""
    target_letters = Counter(normalize_text(target_phrase))
    print("\nLetter counts in target phrase:")
    print("┌───────┬───────┐")
    print("│ Letter│ Count │")
    print("├───────┼───────┤")
    for char, count in sorted(target_letters.items()):
        print(f"│   {char}   │   {count}   │")
    print("└───────┴───────┘")
    print(f"Total: {sum(target_letters.values())} letters\n")

def save_results(results, target_phrase: str):
    """Save results to a file"""
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    filename = f"anagrams-{timestamp}.txt"
    
    with open(filename, 'w') as f:
        f.write(f"Anagrams for: {target_phrase}\n")
        f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        for i, (phrase, fitness, semantic_score, combined_score, unused_count, unused_letters) in enumerate(results, 1):
            # Format the unused letters info
            unused_info = f" (Unused: {', '.join(unused_letters)})" if unused_count > 0 else ""
            
            f.write(f"{i}. {phrase}{unused_info}\n")
            f.write(f"   Fitness: {fitness:.4f}, Semantic: {semantic_score:.1f}, Combined: {combined_score:.1f}\n\n")
    
    print(f"\nResults saved to {filename}")

def print_results(results):
    """Print results to console with formatting"""
    print("\n=== Final Results ===\n")
    
    # Group results by perfect vs near-perfect
    perfect = []
    near_perfect = []
    
    for result in results:
        phrase, fitness, semantic_score, combined_score, unused_count, unused_letters = result
        if unused_count == 0:
            perfect.append(result)
        else:
            near_perfect.append(result)
    
    # Print perfect anagrams first
    if perfect:
        print("PERFECT ANAGRAMS:")
        for i, (phrase, fitness, semantic_score, combined_score, _, _) in enumerate(perfect, 1):
            print(f"{i}. {phrase}")
            print(f"   Combined Score: {combined_score:.1f}, Semantic: {semantic_score:.1f}, Fitness: {fitness:.4f}")
            print()
    
    # Print near-perfect anagrams
    if near_perfect:
        if perfect:
            print("\nNEAR-PERFECT ANAGRAMS:")
        else:
            print("NEAR-PERFECT ANAGRAMS:")
            
        for i, (phrase, fitness, semantic_score, combined_score, unused_count, unused_letters) in enumerate(near_perfect, len(perfect) + 1):
            print(f"{i}. {phrase}")
            print(f"   Combined Score: {combined_score:.1f}, Semantic: {semantic_score:.1f}, Fitness: {fitness:.4f}")
            print(f"   Unused Letters ({unused_count}): {', '.join(unused_letters)}")
            print()
    
    if not perfect and not near_perfect:
        print("No valid anagrams found.")

def main():
    parser = argparse.ArgumentParser(description="LLM-Guided Anagram Generator with Fitness Scoring")
    parser.add_argument("--target", default="WE ARE CLOSED UNTIL FURTHER NOTICE STAY HEALTHY",
                        help="Target phrase to create anagrams from")
    parser.add_argument("--dict", required=True, help="Path to dictionary file")
    parser.add_argument("--api-key", help="Claude API key (if not provided, will try environment or prompt)")
    parser.add_argument("--seeds", type=int, default=20, help="Number of seed words to consider")
    parser.add_argument("--completions", type=int, default=5, 
                        help="Number of completions to request per seed word")
    parser.add_argument("--min-length", type=int, default=1, help="Minimum word length to consider")
    parser.add_argument("--no-two-word", action="store_true", help="Skip two-word algorithm search")
    parser.add_argument("--no-pairs", action="store_true", help="Skip seed word pair search")
    parser.add_argument("--num-pairs", type=int, default=10, help="Number of seed word pairs to try")
    parser.add_argument("--save", action="store_true", help="Save results to a file")
    parser.add_argument("--model", default="claude-3-opus-20240229", 
                        help="Claude model to use (default: claude-3-opus-20240229)")
    parser.add_argument("--perfect-only", action="store_true", 
                        help="Only consider perfect anagrams with no unused letters")
    parser.add_argument("--near-perfect-threshold", type=int, default=4,
                        help="Maximum number of unused letters allowed for near-perfect anagrams")
    
    args = parser.parse_args()
    
    print(f"Target phrase: {args.target}")
    display_letter_counts(args.target)
    print(f"Dictionary: {args.dict}")
    print(f"Using {args.seeds} seed words with {args.completions} completions each.")
    
    if not args.perfect_only:
        print(f"Allowing near-perfect anagrams with up to {args.near_perfect_threshold} unused letters")
    
    # Get API key securely
    api_key = get_api_key(args)
    if not api_key:
        print("Error: No API key provided. Exiting.")
        return
    
    # Generate anagrams
    results = generate_anagrams(
        args.target,
        args.dict,
        api_key,
        num_seeds=args.seeds,
        num_completions=args.completions,
        min_length=args.min_length,
        try_two_word=not args.no_two_word,
        try_pairs=not args.no_pairs,
        num_pairs=args.num_pairs,
        allow_near_perfect=not args.perfect_only,
        near_perfect_threshold=args.near_perfect_threshold
    )
    
    # Print results
    print_results(results)
    
    # Save results if requested
    if results and args.save:
        save_results(results, args.target)

if __name__ == "__main__":
    main()
