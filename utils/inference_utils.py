import numpy as np
import tensorflow as tf


def greedy_search(model, tokenizer, prompt, seq_len, max_gen=20):
    """
    Generate text using greedy decoding

    Args:
        model: Language model
        tokenizer: Tokenizer instance
        prompt: Text prompt
        seq_len: Maximum sequence length
        max_gen: Maximum tokens to generate

    Returns:
        Generated text string
    """
    ids = tokenizer.encode(prompt)
    for _ in range(max_gen):
        # Prepare input
        x = ids[-seq_len:]
        x = np.pad(x, (seq_len - len(x), 0), "constant")

        # Get predictions
        logits = model(tf.constant([x]))

        # Greedy selection
        next_id = int(tf.argmax(logits[0, -1]).numpy())
        ids.append(next_id)

        # Stop if EOS token
        if tokenizer.id2token[next_id] in ["<PAD>", "<EOS>"]:
            break

    return tokenizer.decode(ids)


def top_k_sampling(
    model, tokenizer, prompt, seq_len, k=10, max_gen=20, temperature=1.0
):
    """
    Generate text using top-k sampling

    Args:
        model: Language model
        tokenizer: Tokenizer instance
        prompt: Text prompt
        seq_len: Maximum sequence length
        k: Number of top tokens to consider
        max_gen: Maximum tokens to generate
        temperature: Temperature for sampling

    Returns:
        Generated text string
    """
    ids = tokenizer.encode(prompt)
    for _ in range(max_gen):
        # Prepare input
        x = ids[-seq_len:]
        x = np.pad(x, (seq_len - len(x), 0), "constant")

        # Get predictions
        logits = model(tf.constant([x]))
        logits = logits[0, -1].numpy()

        # Apply temperature
        if temperature != 1.0:
            logits = logits / temperature

        # Top-k filtering
        top_k_ids = logits.argsort()[-k:][::-1]
        probs = np.exp(logits[top_k_ids]) / np.sum(np.exp(logits[top_k_ids]))

        # Sample
        next_id = np.random.choice(top_k_ids, p=probs)
        ids.append(int(next_id))

        # Stop if EOS token
        if tokenizer.id2token[next_id] in ["<PAD>", "<EOS>"]:
            break

    return tokenizer.decode(ids)


def top_p_sampling(
    model, tokenizer, prompt, seq_len, p=0.9, max_gen=20, temperature=1.0
):
    """
    Generate text using top-p (nucleus) sampling

    Args:
        model: Language model
        tokenizer: Tokenizer instance
        prompt: Text prompt
        seq_len: Maximum sequence length
        p: Probability threshold
        max_gen: Maximum tokens to generate
        temperature: Temperature for sampling

    Returns:
        Generated text string
    """
    ids = tokenizer.encode(prompt)
    for _ in range(max_gen):
        # Prepare input
        x = ids[-seq_len:]
        x = np.pad(x, (seq_len - len(x), 0), "constant")

        # Get predictions
        logits = model(tf.constant([x]))
        logits = logits[0, -1].numpy()

        # Apply temperature
        if temperature != 1.0:
            logits = logits / temperature

        # Sort logits in descending order
        sorted_ids = np.argsort(logits)[::-1]
        sorted_probs = np.exp(logits[sorted_ids]) / np.sum(np.exp(logits))

        # Top-p filtering
        cum_probs = np.cumsum(sorted_probs)
        cutoff = np.searchsorted(cum_probs, p) + 1
        top_ids = sorted_ids[:cutoff]
        top_probs = sorted_probs[:cutoff]
        top_probs /= top_probs.sum()

        # Sample
        next_id = np.random.choice(top_ids, p=top_probs)
        ids.append(int(next_id))

        # Stop if EOS token
        if tokenizer.id2token[next_id] in ["<PAD>", "<EOS>"]:
            break

    return tokenizer.decode(ids)


def beam_search(model, tokenizer, prompt, seq_len, beam_width=3, max_gen=20):
    """
    Generate text using beam search

    Args:
        model: Language model
        tokenizer: Tokenizer instance
        prompt: Text prompt
        seq_len: Maximum sequence length
        beam_width: Beam width
        max_gen: Maximum tokens to generate

    Returns:
        Generated text string
    """
    ids = tokenizer.encode(prompt)
    sequences = [(ids, 0.0)]

    for _ in range(max_gen):
        all_candidates = []

        # Expand each current sequence
        for seq, score in sequences:
            # Prepare input
            x = seq[-seq_len:]
            x = np.pad(x, (seq_len - len(x), 0), "constant")

            # Get predictions
            logits = model(tf.constant([x]))
            logits = logits[0, -1].numpy()
            probs = np.exp(logits) / np.sum(np.exp(logits))

            # Create candidates
            for i, p in enumerate(probs):
                # Skip very unlikely tokens
                if p < 1e-5:
                    continue

                candidate = (seq + [i], score - np.log(p + 1e-9))
                all_candidates.append(candidate)

                # Early stopping if EOS token
                if tokenizer.id2token.get(i) in ["<PAD>", "<EOS>"]:
                    candidate = (
                        seq + [i],
                        score - np.log(p + 1e-9) - 10.0,
                    )  # Bonus for completing
                    all_candidates.append(candidate)

        # Order all candidates by score
        ordered = sorted(all_candidates, key=lambda tup: tup[1])

        # Select k best
        sequences = ordered[:beam_width]

        # Stop if best sequence ends with EOS token
        if sequences and tokenizer.id2token.get(sequences[0][0][-1]) in [
            "<PAD>",
            "<EOS>",
        ]:
            break

    # Return best sequence
    best_seq = sequences[0][0]
    return tokenizer.decode(best_seq)
