import numpy as np

# Define the training dataset
data = np.array([
    [1, 1, 0, 1],
    [0, 0, 1, 0],
    [1, 0, 1, 1],
    [0, 1, 0, 0],
    [1, 0, 0, 0],
    [1, 1, 0, 1],
    [0, 1, 1, 1],
    [0, 1, 0, 1],
    [1, 1, 1, 1],
    [1, 0, 1, 0]
])

# Separate features and labels
X = data[:, :-1]  # Features
y = data[:, -1]   # Labels (Spam or Not Spam)

# Calculate prior probabilities
P_spam = np.mean(y)
P_not_spam = 1 - P_spam

# Calculate likelihoods
def calc_likelihood(X, y, feature_idx, label):
    feature_values = X[y == label, feature_idx]
    prob = np.mean(feature_values)
    return prob

# Feature indices
D_idx = 0
U_idx = 1
F_idx = 2

# Calculate likelihoods for Spam
P_D1_given_spam = calc_likelihood(X, y, D_idx, 1)
P_D0_given_spam = 1 - P_D1_given_spam
P_U1_given_spam = calc_likelihood(X, y, U_idx, 1)
P_U0_given_spam = 1 - P_U1_given_spam
P_F1_given_spam = calc_likelihood(X, y, F_idx, 1)
P_F0_given_spam = 1 - P_F1_given_spam

# Calculate likelihoods for Not Spam
P_D1_given_not_spam = calc_likelihood(X, y, D_idx, 0)
P_D0_given_not_spam = 1 - P_D1_given_not_spam
P_U1_given_not_spam = calc_likelihood(X, y, U_idx, 0)
P_U0_given_not_spam = 1 - P_U1_given_not_spam
P_F1_given_not_spam = calc_likelihood(X, y, F_idx, 0)
P_F0_given_not_spam = 1 - P_F1_given_not_spam

# Define new email features
new_email = {'D': 1, 'U': 0, 'F': 0}

# Calculate posterior probabilities
def calc_posterior(P_label, P_D, P_U, P_F, label):
    if label == 1:
        return (P_D if new_email['D'] == 1 else (1 - P_D)) * \
               (P_U if new_email['U'] == 1 else (1 - P_U)) * \
               (P_F if new_email['F'] == 1 else (1 - P_F)) * \
               P_label
    else:
        return (P_D if new_email['D'] == 1 else (1 - P_D)) * \
               (P_U if new_email['U'] == 1 else (1 - P_U)) * \
               (P_F if new_email['F'] == 1 else (1 - P_F)) * \
               P_label

P_spam_given_email = calc_posterior(P_spam, P_D1_given_spam, P_U0_given_spam, P_F0_given_spam, 1)
P_not_spam_given_email = calc_posterior(P_not_spam, P_D1_given_not_spam, P_U0_given_not_spam, P_F0_given_not_spam, 0)

# Print results
print(f"P(Spam | D=1, U=0, F=0) = {P_spam_given_email:.4f}")
print(f"P(Not Spam | D=1, U=0, F=0) = {P_not_spam_given_email:.4f}")

if P_spam_given_email > P_not_spam_given_email:
    print("The email is classified as Spam.")
else:
    print("The email is classified as Not Spam.")
