import argparse
from collections import Counter
import re

def normalize_text(text):
    """Remove non-alphabetic characters and convert to lowercase."""
    return re.sub(r'[^a-z]', '', text.lower())

def validate_anagram(input_str, target_str, verbose=False):
    """Validate if input string is a perfect anagram of target string."""
    # Normalize both strings
    input_clean = normalize_text(input_str)
    target_clean = normalize_text(target_str)
    
    # Create letter frequency counts
    input_counter = Counter(input_clean)
    target_counter = Counter(target_clean)
    
    # Compare counters
    if verbose:
        print(f"Original phrase: '{target_str}'")
        print(f"Candidate anagram: '{input_str}'")
        print("\nLetter counts:")
        
        all_chars = sorted(set(input_counter.keys()) | set(target_counter.keys()))
        print(f"{'Char':<6}{'Target':<8}{'Input':<8}{'Diff':<8}")
        print("-" * 30)
        
        for char in all_chars:
            target_count = target_counter.get(char, 0)
            input_count = input_counter.get(char, 0)
            diff = input_count - target_count
            diff_str = f"{'+' if diff > 0 else ''}{diff}" if diff != 0 else "0"
            print(f"{char:<6}{target_count:<8}{input_count:<8}{diff_str:<8}")
    
    # Calculate difference
    diff = input_counter - target_counter
    extra_chars = {k: v for k, v in diff.items() if v > 0}
    missing_chars = {k: abs(v) for k, v in (target_counter - input_counter).items() if v > 0}
    
    is_valid = input_counter == target_counter
    return is_valid, extra_chars, missing_chars

def main():
    parser = argparse.ArgumentParser(description='Validate if input is a perfect anagram of target phrase')
    parser.add_argument('input', type=str, help='The input string to validate')
    parser.add_argument('--target', type=str, default="WE ARE CLOSED UNTIL FURTHER NOTICE STAY HEALTHY", 
                        help='Target phrase to compare against')
    parser.add_argument('--verbose', '-v', action='store_true', help='Show detailed letter counts')
    
    args = parser.parse_args()
    
    is_valid, extra_chars, missing_chars = validate_anagram(args.input, args.target, args.verbose)
    
    if is_valid:
        print("✅ Valid anagram! Letter counts match exactly.")
    else:
        print("❌ Invalid anagram. Letter frequency mismatch:")
        if extra_chars:
            print("  Extra letters:")
            for char, count in sorted(extra_chars.items()):
                print(f"    '{char}': +{count}")
        if missing_chars:
            print("  Missing letters:")
            for char, count in sorted(missing_chars.items()):
                print(f"    '{char}': -{count}")

if __name__ == "__main__":
    main()
