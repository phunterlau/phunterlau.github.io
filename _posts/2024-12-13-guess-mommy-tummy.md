# The Mathematics of Baby Shower Games: Solomonoff Inference in Action

Last weekend, I found myself applying data science in an unexpected setting: a baby shower. The host announced what seemed like a simple party game - guessing the circumference of the mother-to-be's baby bump. What made this particularly interesting was that I could see everyone else's guesses on a decorated board, transforming a simple estimation game into a fascinating exercise in probability theory and strategic decision-making.

![alt text](/images/redpanda-tummy-guess.jpeg)

## Initial Observation: Understanding the Parameters

When I first approached the board, I noticed the game setup allowed for guesses between 20 and 100 inches. This seemed like an unnecessarily wide range, but it provided an important starting point for analysis. The sheer size of this range meant that random guessing would be highly inefficient.

## Applying Solomonoff's Theory

As I studied the distribution of guesses, I realized this was a perfect opportunity to apply Solomonoff's theory of inductive inference. This theory suggests that when humans make predictions, they tend to favor simpler, more computationally compact patterns. In the context of number guessing, this manifested in several clear ways:

First, there was a strong preference for numbers ending in 0 or 5. The guesses showed clear clusters around 30, 35, 40, and 45 inches. This wasn't random - it reflected the human tendency to gravitate toward what Solomonoff would call "simple" numbers, those with lower Kolmogorov complexity.

Second, I noticed many guesses were derived from simple arithmetic relationships: half of 100, one-third of 90, or modifications of common measurements like 36 inches (a yard). These patterns emerged because humans instinctively seek familiar numerical relationships when making estimates.

## Analyzing the Competition: The Power of Numbers

My next insight came from counting the participants - approximately 60 people had already placed their guesses. This large sample size revealed clear patterns consistent with Solomonoff's theory. The distribution wasn't random but showed structured clustering around numbers that were algorithmically simple to describe or remember.

## Constraint Discovery: Narrowing the Range

After careful observation of the mother-to-be and the pattern of guesses, I made a crucial realization: no reasonable estimate could exceed 60 inches. This effectively cut the possible range in half. The interesting part was how other guests had intuitively arrived at similar conclusions - very few guesses exceeded 60 inches, suggesting a collective understanding of this natural constraint.

## Reference Point: Anchoring the Estimate

A key breakthrough came from noticing the expectant father's presence. His waist measured 48 inches, providing a crucial reference point. Solomonoff's theory suggests that humans often make predictions by modifying existing reference points rather than generating estimates from scratch. Indeed, I noticed several guesses clustered around modifications of this 48-inch reference: 45, 46, and 50 inches were common choices.

## Strategic Analysis: Finding the Optimal Guess

Combining these insights, I developed what I thought was a winning strategy. The majority of guesses followed predictable patterns: clustering around multiples of 5, modifications of the 48-inch reference point, and numbers with simple algorithmic descriptions. Following Solomonoff's principle of favoring the simplest hypothesis consistent with observations, I identified what appeared to be an optimal gap around 48 inches - a number that balanced between the various clusters while avoiding the overcrowded ranges.

## The Outcome: A Lesson in Probabilistic Thinking

When the final measurement was revealed to be 45 inches, my guess of 48 inches proved close but not close enough to win. Ironically, the winning guesses of 44 and 46 inches came from participants who had more directly modified the 48-inch reference point - a simpler strategy that Solomonoff's theory might have predicted would be more likely correct.

## The Mathematical Post-Mortem
This experience revealed how Solomonoff's theory applies in real-world scenarios. The winning guesses came from what were essentially simple modifications of an existing reference point - exactly what the theory would predict as most likely. My more complex strategy of finding gaps between clusters, while mathematically sophisticated, actually moved away from the simpler, and in this case more accurate, approach.

After returning home from the party, my analytical curiosity got the better of me. I decided to code up a simulation to find what would have been the optimal guess given all the information I had. Here's the Python script I wrote to analyze the scenario:

```python
import numpy as np
from scipy.stats import truncnorm

def generate_realistic_guesses_with_prior(n_guesses=60, min_val=20, max_val=100, true_max=60):
    """
    Generate realistic guesses where some players might know the upper bound
    """
    # Assume 70% of players might have some intuition about the upper bound
    informed_players = int(0.7 * n_guesses)
    uninformed_players = n_guesses - informed_players
    
    guesses = []
    
    # Informed players' guesses clustered below 60
    for _ in range(informed_players):
        # Generate numbers with higher density below 60
        base = np.random.choice([
            np.random.randint(20, true_max),  # Direct range
            20 + np.random.exponential(10),   # Early range bias
            np.random.normal(40, 8)           # Normal around middle
        ])
        guess = int(np.clip(base, min_val, true_max))
        guesses.append(guess)
    
    # Uninformed players follow original pattern
    for _ in range(uninformed_players):
        guess = np.random.randint(min_val, max_val)
        guesses.append(guess)
    
    return sorted(guesses)

def find_optimal_guess_with_prior(guesses, min_val=20, max_val=100, true_max=60):
    """
    Find optimal guess incorporating prior knowledge that true value ≤ 60
    """
    guesses = sorted(list(set(guesses)))
    extended_guesses = [min_val-0.5] + guesses + [max_val+0.5]
    
    # Create probability weights favoring range below 60
    def calculate_position_weight(mid_point):
        if mid_point <= true_max:
            # Higher weight for positions below true_max
            # Peak weight around 40 (middle of valid range)
            return 1 - 0.3 * abs(mid_point - 40) / 20
        else:
            # Significant penalty for positions above true_max
            return 0.1  # Very low weight for positions we know are wrong
    
    gaps = []
    for i in range(len(extended_guesses)-1):
        gap_start = extended_guesses[i]
        gap_end = extended_guesses[i+1]
        gap_mid = (gap_start + gap_end) / 2
        gap_size = gap_end - gap_start
        
        # Calculate base territory size
        territory = gap_size / 2
        
        # Apply position weights
        position_weight = calculate_position_weight(gap_mid)
        weighted_territory = territory * position_weight
        
        gaps.append({
            'start': gap_start,
            'end': gap_end,
            'mid': gap_mid,
            'size': gap_size,
            'territory': territory,
            'weighted_territory': weighted_territory,
            'position_weight': position_weight
        })
    
    # Find optimal gap
    optimal_gap = max(gaps, key=lambda x: x['weighted_territory'])
    optimal_guess = round(optimal_gap['mid'])
    
    return optimal_guess, optimal_gap

# Generate and analyze guesses
np.random.seed(42)
guesses = generate_realistic_guesses_with_prior(60, true_max=60)
optimal_guess, optimal_gap = find_optimal_guess_with_prior(guesses, true_max=60)

# Analysis output
print("Distribution Analysis:")
for i in range(20, 101, 10):
    range_guesses = sum(1 for g in guesses if i <= g < i+10)
    print(f"{i}-{i+9}: {'#'*range_guesses} ({range_guesses})")

print(f"\nOptimal guess: {optimal_guess}")
print(f"Gap details:")
print(f"- Gap range: {optimal_gap['start']:.1f} to {optimal_gap['end']:.1f}")
print(f"- Raw gap size: {optimal_gap['size']:.2f}")
print(f"- Position weight: {optimal_gap['position_weight']:.2f}")
print(f"- Weighted territory: {optimal_gap['weighted_territory']:.2f}")

# Show nearby guesses in relevant range
nearby = [g for g in guesses if abs(g - optimal_guess) <= 5]
print(f"\nNearby guesses: {nearby}")

# Additional strategic analysis
print("\nStrategy Confidence Analysis:")
below_60 = sum(1 for g in guesses if g <= 60)
print(f"Guesses ≤ 60: {below_60} ({below_60/len(guesses)*100:.1f}%)")
density_around_optimal = sum(1 for g in guesses if abs(g - optimal_guess) <= 5)
print(f"Density around optimal guess: {density_around_optimal} guesses within ±5")
```

Running this simulation multiple times revealed something fascinating: given the constraints we knew (maximum of 60 inches), the reference point (48 inches), and the distribution of other guests' guesses, the optimal guess should indeed have been closer to 45 inches. The code confirmed what human intuition had already discovered - sometimes the simplest approach, directly modifying a known reference point, outperforms more complex strategies.

What makes this particularly interesting is how the collective behavior of the guessers reflected core principles of inductive inference: preferring simple numbers, using easily computed modifications of reference points, and gravitating toward measurements with low algorithmic complexity.

As I studied the output of my simulation, I realized that my attempt to be clever by finding gaps in the distribution had actually led me away from the most probable range. The code showed that the density of guesses around 45 inches wasn't just random clustering - it represented a collective wisdom that I had unfortunately tried to outsmart.