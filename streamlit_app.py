from functools import partial

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from kde_ebm.mixture_model import fit_all_kde_models, fit_all_gmm_models, get_prob_mat
from kde_ebm.plotting import mixture_model_grid, mcmc_uncert_mat
from kde_ebm.mcmc import mcmc

'''
# Interactive Event-based Model for Disease Progression Modelling

*Currently in development - caching can be improved*

This is an interactive app to play around with the EBM (event-based model), where we can change the parameters of some synthetic data to see how that effects the resulting EBM.

We'll start by preparing the synthetic data.
'''

st.subheader("Data Preparation")

'''
We'll first select the number of controls (healthy) and patients in the dataset:
'''
# Set the seed number to help with caching
seed = st.number_input("RNG Seed", min_value=0, step=1, value=42)
rng = np.random.default_rng(int(seed))

# Select number of controls/patients
left_column, right_column = st.columns(2)
num_controls = left_column.slider("Number of Controls", min_value=100, max_value=1000, value=500, step=100)
num_patients = right_column.slider("Number of Patients", min_value=100, max_value=1000, value=500, step=100)

'''
Now we'll select, for each of 4 hypothetical biomarkers, when in the arbitrary disease timeline pathology begins (the "onset"), and how rapidly the event (i.e. change from normal to abnormal measurement of a biomarker) occurs.
'''

def generate_controls(num_controls):
    return rng.normal(
        0, 0.05,
        size=(num_controls, 4)
    )

def generate_patients(patient_times, sigmoid_func, num_patients, noise):
    x = sigmoid_func(patient_times)

    if noise > 0:
        x += rng.normal(
            0,
            noise,
            num_patients
        )
    return x

def sigmoid(x, b, a=1):
    return 1/(1 + np.exp(-a*(x-b)))

@st.cache
def generate_data(num_controls, num_patients, patient_times, onsets, grads, noise):
    # Create container for patients
    patients = np.full((num_patients, 4), np.nan)
    # Container for the biomarker functions
    sigmoid_funcs = []
    # Loop through sigmoids
    for i, (onset, grad) in enumerate(zip(onsets, grads)):
        # Create func for this sigmoid
        sigmoid_funcs.append(
            partial(sigmoid, a=grad, b=onset)
        )
        # Generate data for biomarker
        patients[:, i] = generate_patients(
            patient_times, sigmoid_funcs[i], num_patients, noise
        )
    # Generate control data
    controls = generate_controls(num_controls)
    # Combine and return
    data = np.vstack([
        patients,
        controls
    ])
    return data, sigmoid_funcs

# Cols for biomarker onsets
col_ll, col_lr, col_rl, col_rr = st.columns(4)

MAX_TIME = 30

# Ranges for biomarker onsets
onset_ranges = {
    "min_value": 0,
    "max_value": MAX_TIME,
    "step": 1
}
grad_ranges = {
    "min_value": 0.0,
    "max_value": 1.0,
    "step": 0.05
}

init_value = MAX_TIME / 5

# Sliders for biomarker pathology onset
onset1 = col_ll.slider(
    "Biomarker 1 Onset",
    value=int(init_value),
    **onset_ranges
)
grad1 = col_lr.slider(
    "Biomarker 1 Gradient",
    value=1.0,
    **grad_ranges
)
onset2 = col_ll.slider(
    "Biomarker 2 Onset",
    value=int(init_value)*2,
    **onset_ranges
)
grad2 = col_lr.slider(
    "Biomarker 2 Gradient",
    value=1.0,
    **grad_ranges
)
onset3 = col_rl.slider(
    "Biomarker 3 Onset",
    value=int(init_value)*3,
    **onset_ranges
)
grad3 = col_rr.slider(
    "Biomarker 3 Gradient",
    value=1.0,
    **grad_ranges
)
onset4 = col_rl.slider(
    "Biomarker 4 Onset",
    value=int(init_value)*4,
    **onset_ranges
)
grad4 = col_rr.slider(
    "Biomarker 4 Gradient",
    value=1.0,
    **grad_ranges
)

onsets = [onset1, onset2, onset3, onset4]
grads = [grad1, grad2, grad3, grad4]
# Whether noise is added to the patients
noise = st.slider("Noise added to patients:", min_value=0.0, max_value=0.5, step=0.05, value=0.1)

# Get all the x-values
patient_times = rng.uniform(
    onset_ranges["min_value"],
    MAX_TIME,
    num_patients
)

# Generate the data
data, sigmoid_funcs = generate_data(num_controls, num_patients, patient_times, onsets, grads, noise)

'''
### Data Visualization

Let's take a look at what our synthetic data looks like. You can either look at the true (sigmoid) functions and the sampled data, or histograms of the biomarkers colourised by whether they are patients or controls.
'''

def plot_sigmoids(data, num_patients, num_controls, sigmoid_funcs, colors):
    fig, ax = plt.subplots()

    # Loop over events/sigmoids
    for i, sig_func in enumerate(sigmoid_funcs):
        # Select the colour
        c = np.array(colors[i])
        # Plot the sigmoid line
        ax.plot(
            np.linspace(0, MAX_TIME, num_patients),
            sig_func(np.linspace(0, MAX_TIME, num_patients)),
            color=c,
            linestyle="solid",
            linewidth=3,
            label=f"Biomarker {i+1}"
        )
        # Plot the patients
        ax.scatter(
            patient_times,
            data[:num_patients, i],
            marker="o",
            s=10,
            color=c
        )
        # Plot the controls
        ax.scatter(
            np.zeros(data[num_patients:, i].shape),
            data[num_patients:, i],
            marker="v",
            s=10,
            color=c
        )
    ax.set_xlabel("Hypothetical Disease Time")
    ax.set_ylabel("Sigmoid (t)")
    ax.legend(loc=(1.05, 0.765))
    return fig, ax


def plot_histograms(df, num_patients, num_controls):
    fig, axs = plt.subplots(2, 2)

    axs = axs.flatten()

    for i, ax in enumerate(axs):
        if i != 0:
            sns.histplot(
                data=df,
                x=f"Biomarker {i+1}",
                hue="Labels",
                bins=50,
                legend=False,
                ax=ax
            )
        else:
            sns.histplot(
                data=df,
                x=f"Biomarker {i+1}",
                hue="Labels",
                bins=50,
                legend=True,
                ax=ax
            )
    fig.tight_layout()
    return fig, axs

# Convert data to a DataFrame
df = pd.DataFrame(
    data,
    columns=[f"Biomarker {i+1}" for i in range(data.shape[1])]
)
# Add labels
df["Labels"] = ["Patient" for i in range(num_patients)] + ["Control" for i in range(num_controls)]

# Selection for visualizing the synthetic data
data_viz = st.radio(
    "",
    ("Sigmoids", "Histograms")
)

# Colour palette
colors = sns.color_palette("colorblind")

# Select visualization method
if data_viz == "Sigmoids":
    fig_viz, ax = plot_sigmoids(data, num_patients, num_controls, sigmoid_funcs, colors)

elif data_viz == "Histograms":
    fig_viz, axs = plot_histograms(df, num_patients, num_controls)

st.pyplot(fig=fig_viz)


'''
### Event-Based Model
'''

# @st.cache
def fit_ebm(data, labels, ebm_method):
    # Positive means worsening
    disease_direction = [1, 1, 1, 1]
    # Select EBM method
    if ebm_method == "GMM":
        mixtures = fit_all_gmm_models(
            data, labels,
            implement_fixed_controls=True
        )
    elif ebm_method == "KDE":
        mixtures = fit_all_kde_models(
            data, labels,
            implement_fixed_controls=True,
            patholog_dirn_array=disease_direction
        )

    return mixtures

st.write("We'll first select the EBM method to use: Gaussian mixture model (GMM) or kernel density estimation (KDE).")
ebm_method = st.radio(
    "EBM Method:",
    ("GMM", "KDE")
)
st.write("Then we'll use `kde_ebm.plotting.mixture_model_grid` to look at the biomarkers and the GMMs/KDEs that we've fit to identify what value we can consider abnormal, and thus the disease event has occurred.")

labels = df["Labels"].map({"Patient": 1, "Control": 0}).values
data = df.drop("Labels", axis=1).values

mixtures = fit_ebm(data, labels, ebm_method)

fig_mm, ax = mixture_model_grid(
    data, labels,
    mixtures,
    score_names=[f"Biomarker {i+1}" for i in range(data.shape[1])],
    class_names=['Controls', 'Patients']
)

st.pyplot(fig=fig_mm)

'''
With these mixture models for each biomarker, we can find the maximum likelihood sequence via MCMC (markov chain monte carlo), and then plot this on a positional variance diagram, to illustrate the certainty of this sequence (across the MCMC samples).
'''

@st.cache
def run_mcmc(data, mixtures):
    with st.spinner("Running MCMC..."):
        mcmc_samples = mcmc(data, mixtures, plot=False, greedy_n_iter=1000, n_iter=10000)
    num_events = data.shape[1]
    # Get all MCMC orderings
    all_orders = np.array([x.ordering for x in mcmc_samples])
    # Calculate certainty
    confusion_mat = np.sum(
        all_orders[:, :, None] == np.arange(num_events),
        axis=0
    )
    # Normalize
    confusion_mat = confusion_mat / all_orders.shape[0]
    # breakpoint()
    return confusion_mat, mcmc_samples

num_events = data.shape[1]

confusion_mat, mcmc_samples = run_mcmc(data, mixtures)

# Plot the confusion matrix
fig_pvd, ax = plt.subplots()
ax.imshow(confusion_mat, cmap="Oranges", vmin=0, vmax=1)
ax.set_xticks(np.arange(num_events))
ax.set_xticklabels(np.arange(1, num_events+1))
ax.set_yticks(np.arange(num_events))
ax.set_yticklabels([f"Biomarker {i+1}" for i in range(num_events)])
ax.set_xlabel("Event Order")

# mcmc_samples = mcmc(data, mixtures)
# print(mcmc_samples)
# fig_pvd, ax = mcmc_uncert_mat(mcmc_samples, ml_order=mcmc_samples[0], score_names=[f"Biomarker {i+1}" for i in range(data.shape[1])])

st.pyplot(fig=fig_pvd)

'''
#### Patient Staging
With our model, we can then assign (for each individual) a disease stage. We'll plot this for the controls and patients below, where (ideally) the controls should all be at stage 0 (i.e. no disease event has occurred) and the patients are distributed across the events.

As we adjust the onset for each biomarker, the proportions of the events should change. In particular, as biomarkers get closer in "time" for disease onset more individuals will concenrate around those stages.

A scatterplot can also be shown, where the colour indicates the stage assigned which should largely follow the generated sigmoids we saw at the start.
'''

@st.cache
def assign_stages(df, mixtures, mcmc_samples):
    data = df.drop("Labels", axis=1).values
    # Get probability matrix for normal/abnormal
    prob_mat = get_prob_mat(data, mixtures)
    
    stages, stage_likelihoods = mcmc_samples[0].stage_data(prob_mat)
    # Calc expected stages
    expected_stages = (stage_likelihoods.T * np.arange(1,stage_likelihoods.shape[1]+1)[:, None]).T.sum(1) / stage_likelihoods.sum(1) - 1
    # breakpoint()

    df["Stage"] = stages
    return df

df = assign_stages(df, mixtures, mcmc_samples)

# Selection for visualizing the synthetic data
stages_viz = st.radio(
    "",
    ("Histogram", "Scatterplot")
)

fig_stages, ax = plt.subplots()

# Select visualization method
if stages_viz == "Histogram":
    sns.histplot(
        data=df,
        x="Stage",
        hue="Labels",
        bins=50,
        legend=True,
        discrete=True,
        ax=ax
    )

elif stages_viz == "Scatterplot":
    # breakpoint()
    # sns.scatterplot(
    #     data=df.melt(value_vars=[f"Biomarker {i+1}" for i in range(data.shape[1])], var_name="Biomarker", value_name="Sigmoid (t)"),
    #     x=np.hstack([np.repeat(patient_times, 4), np.zeros((num_controls*4,))]),
    #     y="Sigmoid (t)",
    #     hue="Biomarker",
    #     palette="viridis",
    #     ax=ax
    # )

    for biomarker in range(4):
        ax.scatter(
            patient_times,
            data[:num_patients, biomarker],
            c=df["Stage"].values[:num_patients],
            cmap="viridis",
            s=10,
        )

    ax.set_xlabel("Hypothetical Disease Time")
    ax.set_ylabel("Sigmoid (t)")

st.pyplot(fig=fig_stages)