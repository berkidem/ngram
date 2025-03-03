import streamlit as st
import numpy as np
import pickle
import os
import re
import ast

# -----------------------------------------------------------------------------
# Set up page configuration
st.set_page_config(
    page_title="Occupation Sequence Generator", page_icon="👨‍💼", layout="wide"
)


# -----------------------------------------------------------------------------
# Define a utility function to get the logs from the model and converts that into user-friendly format
def convert_log(log, token_to_char):
    """
    Searches for a token list in the log string and replaces it with the actual occupation names.
    """
    pattern = r"\[(.*?)\]"
    match = re.search(pattern, log)
    if match:
        token_str = match.group(0)
        try:
            # Convert the token list string to a Python list
            tokens = ast.literal_eval(token_str)
            # Convert each token to its occupation name using token_to_char mapping
            occupation_names = [
                token_to_char.get(token, str(token)) for token in tokens
            ]
            new_token_str = "[" + ", ".join(occupation_names) + "]"
            # Replace the original token list in the log with the occupation names
            new_log = log.replace(token_str, new_token_str)
            return new_log
        except Exception as e:
            # If conversion fails, return the original log
            return log
    return log


# -----------------------------------------------------------------------------


def plot_top20(values, token_to_char, chosen_token, title):
    """
    Plot a bar chart of the top 20 values (counts or probabilities) sorted in descending order.
    If the chosen token is not in the top 20, it is appended and highlighted in red.
    """
    import pandas as pd
    import matplotlib.pyplot as plt

    # Create a Pandas Series, sorted descending
    series = pd.Series(values, index=[token_to_char[i] for i in range(len(values))])
    series = series.sort_values(ascending=False)

    # Select the top 20 (plus chosen token, if missing)
    top20 = series.head(20)
    chosen_occ = token_to_char[chosen_token]
    if chosen_occ not in top20.index:
        top20 = pd.concat([top20, pd.Series({chosen_occ: series[chosen_occ]})])

    # Prepare color coding for chosen token
    colors = ["red" if occ == chosen_occ else "blue" for occ in top20.index]

    # Create the figure with increased size
    fig, ax = plt.subplots(figsize=(12, 5))

    # Plot the bar chart
    ax.bar(range(len(top20)), top20.values, color=colors)

    # Set title and y-axis label
    ax.set_title(title)
    ax.set_ylabel("Value")

    # Set custom x-ticks and x-tick labels
    ax.set_xticks(range(len(top20)))
    ax.set_xticklabels(top20.index, rotation=60, ha="right")

    # Adjust layout to reduce overlap
    fig.tight_layout()
    return fig


def plot_sampling_intervals(prob_values, chosen_token, coinf, token_to_char, title):
    """
    Plot a horizontal bar from 0 to 1, subdivided by each token's probability (top 20 + chosen token).
    A vertical line at coinf shows the random number in [0, 1].
    Whichever token segment covers coinf is the chosen token.

    Labels:
    - We label the top few tokens plus the chosen token if not already in the top few.
    - Labels appear above the bar, horizontally, in black text for visibility.
    """
    import pandas as pd
    import matplotlib.pyplot as plt

    # Convert probabilities to a Series and sort descending
    series = pd.Series(
        prob_values, index=[token_to_char[i] for i in range(len(prob_values))]
    )
    series = series.sort_values(ascending=False)

    # Keep top 20 plus the chosen token if missing
    top20 = series.head(20)
    chosen_occ = token_to_char[chosen_token]
    if chosen_occ not in top20.index:
        top20 = pd.concat([top20, pd.Series({chosen_occ: series[chosen_occ]})])

    tokens_ordered = top20.index
    probs_ordered = top20.values

    # Create a figure that's wide and tall enough for the bar plus labels
    fig, ax = plt.subplots(figsize=(10, 3))

    left_edge = 0.0
    # Label the top N_LABELS tokens plus the chosen token
    N_LABELS = 3
    label_set = set(tokens_ordered[:N_LABELS])
    if chosen_occ not in label_set:
        label_set.add(chosen_occ)

    # Draw stacked horizontal segments
    for token_name, p in zip(tokens_ordered, probs_ordered):
        right_edge = left_edge + p

        # Color the segment red if it contains coinf, otherwise blue
        if left_edge <= coinf < right_edge:
            color = "red"
        else:
            color = "blue"

        # Draw the bar segment
        ax.barh(
            y=0,
            width=p,
            left=left_edge,
            height=0.4,
            color=color,
            edgecolor="black",
            linewidth=1,
            alpha=0.7,
        )

        # If this token is in label_set, place a label above its segment
        if token_name in label_set:
            midpoint = (left_edge + right_edge) / 2
            ax.text(
                x=midpoint,
                y=0.45,  # Slightly above the bar
                s=token_name,
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=0,  # Horizontal text
                color="black",
            )

        left_edge = right_edge

    # Draw a vertical dashed line at coinf
    ax.axvline(coinf, color="black", linestyle="--")

    # Configure the axes
    ax.set_xlim(0, 1)
    # Increase top y-limit so labels fit above the bar
    ax.set_ylim(-0.2, 1.0)
    ax.set_yticks([])
    ax.set_xlabel("Probability Interval")
    ax.set_title(title)

    fig.tight_layout()
    return fig


# -----------------------------------------------------------------------------
# Define the RNG and sampling functions from the original code


class RNG:
    def __init__(self, seed):
        self.state = seed

    def random_u32(self):
        self.state ^= (self.state >> 12) & 0xFFFFFFFFFFFFFFFF
        self.state ^= (self.state << 25) & 0xFFFFFFFFFFFFFFFF
        self.state ^= (self.state >> 27) & 0xFFFFFFFFFFFFFFFF
        return ((self.state * 0x2545F4914F6CDD1D) >> 32) & 0xFFFFFFFF

    def random(self):
        # random float32 in [0, 1)
        return (self.random_u32() >> 8) / 16777216.0


def sample_discrete(probs, coinf):
    # sample from a discrete distribution
    cdf = 0.0
    for i, prob in enumerate(probs):
        cdf += prob
        if coinf < cdf:
            return i
    return len(probs) - 1  # in case of rounding errors


# -----------------------------------------------------------------------------
# Define the n-gram model class


class NgramModel:
    def __init__(self, vocab_size, seq_len, smoothing=0.0):
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.smoothing = smoothing
        # the parameters of this model: an n-dimensional array of counts
        self.counts = np.zeros((vocab_size,) * seq_len, dtype=np.uint32)
        # a buffer to store the uniform distribution, just to avoid creating it every time
        self.uniform = np.ones(self.vocab_size, dtype=np.float32) / self.vocab_size

    def train(self, tape):
        assert isinstance(tape, list)
        assert len(tape) == self.seq_len
        self.counts[tuple(tape)] += 1

    def get_counts(self, tape):
        assert isinstance(tape, list)
        assert len(tape) == self.seq_len - 1
        return self.counts[tuple(tape)]

    def __call__(self, tape):
        # returns the conditional probability distribution of the next token
        assert isinstance(tape, list)
        assert len(tape) == self.seq_len - 1
        # get the counts, apply smoothing, and normalize to get the probabilities
        counts = self.counts[tuple(tape)].astype(np.float32)
        counts += self.smoothing  # add smoothing ("fake counts") to all counts
        counts_sum = counts.sum()
        probs = counts / counts_sum if counts_sum > 0 else self.uniform
        return probs

    def load_counts(self, counts):
        assert counts.shape == self.counts.shape
        self.counts = counts


# -----------------------------------------------------------------------------
# Define the BackoffNgramModel class


class BackoffNgramModel:
    """
    A backoff model that can be used to combine multiple n-gram models of different orders.
    During inference, it uses the highest order model that has sufficient data for the current context.
    """

    def __init__(self, vocab_size, seq_lens, smoothings, counts_threshold=0):
        self.vocab_size = vocab_size
        self.counts_threshold = counts_threshold
        self.models = {}
        self.logs = []

        # Initialize models for each order
        for i, (seq_len, smoothing) in enumerate(zip(seq_lens, smoothings)):
            self.models[seq_len] = NgramModel(vocab_size, seq_len, smoothing)

        # Sort the sequence lengths in descending order for backoff
        self.seq_lens = sorted(seq_lens, reverse=True)

    # def __call__(self, tape):
    #     # Find the highest order model that has sufficient data for the current context
    #     for seq_len in self.seq_lens:
    #         # Adjust tape length for the current model
    #         context_len = seq_len - 1
    #         if len(tape) >= context_len:
    #             context = tape[-context_len:] if context_len > 0 else []

    #             # Get counts for the current context
    #             counts = self.models[seq_len].get_counts(context)
    #             if counts.sum() > self.counts_threshold:
    #                 return self.models[seq_len](context)

    #     # If no model has sufficient data, use unigram model
    #     return self.models[1]([])

    def __call__(self, tape):
        # Find the highest order model that has sufficient data for the current context
        for seq_len in self.seq_lens:
            context_len = seq_len - 1
            if len(tape) >= context_len:
                context = tape[-context_len:] if context_len > 0 else []
                counts = self.models[seq_len].get_counts(context)
                if counts.sum() > self.counts_threshold:
                    # Logging the selected model order, context, and count for debugging
                    print(
                        f"Using {seq_len}-gram model with context: {context}, count: {counts.sum()}"
                    )
                    self.logs.append(
                        f"Using {seq_len}-gram model with context {context} (count: {counts.sum()})"
                    )

                    return self.models[seq_len](context)

        # If no model has sufficient data, log the fallback and use unigram model
        print(
            "Falling back to unigram model due to insufficient data in higher-order models."
        )
        self.logs.append(
            "Falling back to unigram model due to insufficient data in higher-order models."
        )
        return self.models[1]([])

    def load_counts(self, counts_dict):
        # Load counts for each model
        for seq_len, counts in counts_dict.items():
            if seq_len in self.models:
                self.models[seq_len].load_counts(counts)


# -----------------------------------------------------------------------------
# Load data and create models


@st.cache_resource
def load_data():
    # Load token mappings
    try:
        with open("../data/onet_name_encoding.pkl", "rb") as f:
            encoding = pickle.load(f)
            char_to_token = encoding
            token_to_char = {v: k for k, v in char_to_token.items()}
    except:
        # Fallback if pickle format not available
        import pandas as pd

        encoding = pd.read_parquet("../data/onet_name_encoding.parquet")
        char_to_token = encoding.set_index("BGI_ONET_NAME")[
            "BGI_ONET_NAME_ENCODED"
        ].to_dict()
        token_to_char = {v: k for k, v in char_to_token.items()}

    # Define EOT token
    EOT_TOKEN = 1016
    token_to_char[EOT_TOKEN] = "<END_OF_TEXT>"
    char_to_token["<END_OF_TEXT>"] = EOT_TOKEN

    # Load n-gram counts
    counts_dict = {}
    for seq_len in [3, 2, 1]:
        try:
            counts_path = os.path.join("../dev", f"ngram_raw_counts_{seq_len}_gram.npy")
            counts = np.load(counts_path)
            counts_dict[seq_len] = counts
        except:
            st.error(
                f"Could not load counts for {seq_len}-gram model. Make sure the file exists at {counts_path}"
            )

    vocab_size = 1017  # 0-1016 inclusive

    return char_to_token, token_to_char, counts_dict, vocab_size, EOT_TOKEN


# Load data
char_to_token, token_to_char, counts_dict, vocab_size, EOT_TOKEN = load_data()

# -----------------------------------------------------------------------------
# Create the Streamlit app interface

st.title("Occupation Sequence Generator")
st.write(
    """
This app generates occupation sequences using a backoff n-gram "language" model.

**This is an extremely simple model and it does not generate realistic occupation sequences.** However, it is a fun way to explore our data from a new angle. Moreover, developing this model makes is much easier to train significantly more advanced models, since all the building blocks are already in place and we just need to change the model. When I get a chance to develop a more advanced model, I will add it here as an option or made a similar app for it so that we can compare the results.

Enter the first k occupations in a career, and the app will generate the next l occupations.
"""
)

# Sidebar for model parameters
st.sidebar.header("Model Parameters")
counts_threshold = st.sidebar.slider("Minimum counts threshold for backoff", 0, 100, 5)
smoothing_3gram = st.sidebar.slider("3-gram smoothing", 0.01, 10.0, 0.3, step=0.01)
smoothing_2gram = st.sidebar.slider("2-gram smoothing", 0.01, 10.0, 0.5, step=0.01)
smoothing_1gram = st.sidebar.slider("1-gram smoothing", 0.01, 10.0, 1.0, step=0.01)
random_seed = st.sidebar.number_input(
    "Random seed", value=1337, min_value=0, max_value=9999
)
visualize_probs = st.sidebar.checkbox("Show Probability Visuals", value=False)

# Input parameters
col1, col2 = st.columns(2)
with col1:
    k = st.number_input(
        "Number of initial occupations (k)", min_value=0, max_value=5, value=1
    )
with col2:
    l = st.number_input(
        "Number of occupations to generate (l)", min_value=1, max_value=10, value=3
    )

# Create occupation selection widgets based on k
occupation_inputs = []
available_occupations = sorted(
    [
        token_to_char[i]
        for i in range(vocab_size)
        if i != EOT_TOKEN and i in token_to_char
    ]
)

for i in range(k):
    occupation = st.selectbox(
        f"Occupation {i+1}", options=available_occupations, key=f"occupation_{i}"
    )
    occupation_inputs.append(occupation)

if st.button("Generate Sequence"):
    # Initialize the backoff model
    backoff_model = BackoffNgramModel(
        vocab_size=vocab_size,
        seq_lens=[3, 2, 1],
        smoothings=[smoothing_3gram, smoothing_2gram, smoothing_1gram],
        counts_threshold=counts_threshold,
    )
    backoff_model.load_counts(counts_dict)
    sample_rng = RNG(random_seed)

    # Prepare initial tape from selected occupations (or use EOT_TOKEN if none)
    tape = [char_to_token[occ] for occ in occupation_inputs] if k > 0 else [EOT_TOKEN]
    generated_tokens = []

    st.header("Inference Process Visualization")

    # Run inference for l steps
    for step in range(l):
        st.subheader(f"Step {step+1}")

        # determine the model to use based on the current tape
        chosen_order = None
        chosen_context = None
        for seq_len in sorted([3, 2, 1], reverse=True):
            context_len = seq_len - 1
            if len(tape) >= context_len:
                context = tape[-context_len:] if context_len > 0 else []
                raw_counts = backoff_model.models[seq_len].get_counts(context)
                if raw_counts.sum() > counts_threshold:
                    chosen_order = seq_len
                    chosen_context = context
                    backoff_model.logs.append(
                        f"Using {seq_len}-gram model with context {context} (raw count sum: {raw_counts.sum()})"
                    )
                    break
        if chosen_order is None:
            chosen_order = 1
            chosen_context = []
            backoff_model.logs.append(
                "Falling back to unigram model due to insufficient data in higher-order models."
            )

        chosen_model = backoff_model.models[chosen_order]
        # Compute probabilities with smoothing using the chosen model
        probs = chosen_model(chosen_context)

        # Retrieve raw counts (without smoothing)
        raw_counts = chosen_model.get_counts(chosen_context).astype(np.float32)
        counts_sum = raw_counts.sum()
        raw_probs = raw_counts / counts_sum if counts_sum > 0 else chosen_model.uniform

        # Compute smoothed probabilities manually
        smoothed_counts = raw_counts + chosen_model.smoothing
        smoothed_probs = (
            smoothed_counts / smoothed_counts.sum()
            if smoothed_counts.sum() > 0
            else chosen_model.uniform
        )

        # Sample a token using smoothed probabilities
        coinf = sample_rng.random()
        chosen_token = sample_discrete(smoothed_probs.tolist(), coinf)

        st.markdown(
            f"**Model used:** {chosen_order}-gram with context: {chosen_context}"
        )

        if visualize_probs:
            # Plot the three visualizations in three columns
            col1, col2, col3 = st.columns(3)
            with col1:
                fig1 = plot_top20(
                    raw_counts, token_to_char, chosen_token, "Raw Counts (Top 20)"
                )
                st.pyplot(fig1)
            with col2:
                fig2 = plot_top20(
                    raw_probs, token_to_char, chosen_token, "Raw Probabilities (Top 20)"
                )
                st.pyplot(fig2)
            with col3:
                fig3 = plot_top20(
                    smoothed_probs,
                    token_to_char,
                    chosen_token,
                    "Smoothed Probabilities (Top 20)",
                )
                st.pyplot(fig3)

        st.write(f"Chosen token: **{token_to_char[chosen_token]}**")
        # Append the chosen token for the next step
        tape.append(chosen_token)
        generated_tokens.append(chosen_token)

        # After sampling chosen_token
        if visualize_probs:
            fig_sampling = plot_sampling_intervals(
                smoothed_probs,
                chosen_token,
                coinf,
                token_to_char,
                "Sampling from [0,1]",
            )
            st.pyplot(fig_sampling)

    st.header("Final Generated Sequence")
    full_sequence = occupation_inputs + [token_to_char[t] for t in generated_tokens]
    st.write(" → ".join(full_sequence))

    if visualize_probs:
        st.markdown(
            """
    ### About the Probability Visualizations

    When **Show Probability Visuals** is enabled in the sidebar, the app displays three bar charts at each generation step:

    1. **Raw Counts (Top 20)**: Shows how frequently each occupation (token) has appeared in the training data for the given context, **without any smoothing**.  
    2. **Raw Probabilities (Top 20)**: Normalizes the raw counts into probabilities, i.e., dividing each token's count by the sum of all counts.  
    3. **Smoothed Probabilities (Top 20)**: Applies smoothing to the counts (adds a small positive constant) before normalizing, ensuring no token has exactly zero probability.

    In all three charts, the chosen token is highlighted in red. If the chosen token is not in the top 20 by probability, it is appended to the chart so you can still see it.

    **Sampling from [0,1]**: The additional horizontal bar shows how a single random number in [0,1] determines which token is selected. Each token occupies a segment of the 0–1 interval proportional to its probability; the random number is shown as a dashed line. Whichever segment includes that line is the chosen token.

    By comparing these visuals, you can see how raw counts transform into probabilities, how smoothing adjusts those probabilities, and exactly how a single random draw picks a particular occupation.

    To learn more about what any of these terms mean, read on below!
    """
        )

    st.markdown("### Debug Logs")
    for log in backoff_model.logs:
        st.write(log)


# Add a section explaining the backoff model
st.markdown(
    """
## About the Backoff N-gram Model

N-gram language models are a type of probabilistic model that predicts the next item in a sequence based on the previous $n-1$ items. In fact, n-gram is simply a fancy name for a Markov chain where the state is the last $n-1$ items. For example, a trigram model predicts the next item based on the previous two items. Similarly, a bigram model predicts the next item based on the previous item and a unigram model predicts the next item based on unconditioned probabilities; it is the trivial Markov chain.

There are two components of the models presented in this app that differs from the traditional Markov chain:

**Smoothing**: This is a technique used to handle unseen events. In the context of language models, it assigns a small probability to unseen n-grams. These models are trained using training data and evaluated on validation and test data, just like any other supervised learning model. Here is the purpose of smoothing: If we have a trigram model, it is possible that there are some sequences that don't appear in the training data but appears in the validation and test data. If our model was purely Markovian, it would assign 0 probability to such a chain, since the model has never seen it in its training. However, this might make the model too conversative and "closed-minded". Moreover, if we try to condition on a previously unseen sequence, we would be dividing by zero. That would mean the perplexity of the model would be through the roof. We don't want that."""
)

cols = st.columns([1, 2, 1])
with cols[1]:
    st.image(
        "https://allears.net/wp-content/uploads/2020/08/im-perplexed-hamilton-gif.gif",
        caption="When a model encounters a sequence it's never seen before",
    )


st.markdown(
    """So, we pretend each possible context sequence was seen with a very small frequency. (So we simply add a very small number to each count, to make the probability of every event positive.) Thus, smoothing helps the model generalize better to unseen sequences; otherwise it would throw a tantrum everytime we see a previously unseen sequence. N-gram smoothing parameters that can be controled from the sidebar determine what "a very small frequency" mean for each model. (More on this below.)

**Backoff**: This is a technique used to handle sparse data. In the context of language models, it allows the model to "back off" to lower-order models when there isn't enough data to support the current order. For example, if the trigram model has seen only one observation with a particular context sequence, its prediction will not be very reliable. Instead of trusting that single observation, the model can "back off" to the bigram model, and so on. This allows the model to make optimally limit its context window when the full context window is not very informative.

For instance, if someone was a "Shampooer", "Mathematician" and "Data Scientist", it is entirely possible that the fact that this person was a "Shampooer" is not very informative about what job they get after their "Data Scientist" position. So, the model should not give too much weight to the first occupation. This is the purpose of backoff: To make the model more flexible and less dogmatic about the context window. When a model should back off is determined by the minimum counts threshold parameter that can be controlled from the sidebar.

In this app, the backoff model works by trying to use the highest-order n-gram model (trigram) first. If there isn't enough data in the trigram model (determined by the counts threshold), 
it "backs off" to a lower-order model (bigram, then unigram).

## Summary of Parameters

The minimum counts threshold controls when the model decides to back off to a lower-order model:
- Higher threshold: More conservative, backs off more frequently, relies more on lower-order models
- Lower threshold: More aggressive, tries to use higher-order models even with sparse data

The smoothing parameters control how much probability mass is allocated to unseen events:
- Higher smoothing: More probability for unseen sequences, more diverse generations but it can become too generic
- Lower smoothing: Sticks closer to observed patterns in training data, more grounded generations but can be too conservative

Finally, to get different sequences from the same input, you can change the random seed. Otherwise, the same input will always generate the same output.

Feel free to play around with these parameters to see how they affect the generated sequences!

And, one again, **this is not a serious model**. It is just a fun way to explore the data. Please use responsibly!
"""
)

st.markdown("---")
st.caption(
    "Developed by [berkidem](https://github.com/berkidem). The model underlying this app was forked from [Karpathy's ngram language model](https://github.com/karpathy/ngram) and trained on Burning Glass Institute's profiles data. [GitHub repo](https://github.com/berkidem/ngram)."
)
