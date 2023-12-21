from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import defaultdict
import string
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu, chi2_contingency
import numpy as np
from scipy import stats
from scipy.stats import ttest_ind

## Download required NLTK packages
# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download('wordnet')

df = pd.read_csv('BOM.csv')


# Function to find hapax legomena
def find_hapax_legomena(text):
    # Tokenize the text
    token_text = word_tokenize(text)
    # Convert to lower case and remove punctuation and stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word.lower() for word in token_text if word.isalpha() and word.lower() not in stop_words]
    # Count the words
    word_counts = Counter(tokens)
    # Identify hapax legomena
    hapaxes = [word for word, count in word_counts.items() if count == 1]
    return hapaxes

combined_text = ' '.join(df['text'])
hapaxes = find_hapax_legomena(combined_text)

def find_hapaxes_in_verse(verse, hapaxes):
    found_hapaxes = []
    for hapax in hapaxes:
        if hapax in verse.lower():
            found_hapaxes.append(hapax)
    return found_hapaxes


def find_all_hapax_details(df, hapaxes):
    """
    Find the verse, event number, and event_written status for all hapax legomena.
    Ensures precise matching of hapaxes in the text.
    """
    hapax_details = []
    for index, row in df.iterrows():
        # Tokenize the verse text
        verse_tokens = word_tokenize(row['text'].lower())
        verse_hapaxes = set(hapax for hapax in hapaxes if hapax in verse_tokens)
        for hapax in verse_hapaxes:
            hapax_details.append({
                'hapax': hapax,
                'verse': row['verse_title'],  # Assuming 'verse_title' is the column for verse
                'event': row['event'],       # Assuming 'event' is the column for event number
                'event_written': row['event_written']
            })
    return hapax_details

# Then use this function to find details for all hapaxes
all_hapax_details = find_all_hapax_details(df, hapaxes)

# Create a DataFrame from the details of all hapaxes
all_hapax_details_df = pd.DataFrame(all_hapax_details)

# Count the number of hapaxes in each event
event_hapax_counts = all_hapax_details_df.groupby(['event', 'event_written']).size().reset_index(name='hapax_count')

# Create a boxplot for the event_hapax_counts DataFrame
plt.figure(figsize=(10, 6))
sns.boxplot(x='event_written', y='hapax_count', data=event_hapax_counts)
plt.title('Boxplot of Hapax Legomena Counts per Event')
plt.xlabel('Event Written Status (1: Written, 0: Non-Written)')
plt.ylabel('Number of Hapax Legomena per Event')
plt.show()


# # Histogram of Hapax Legomena Counts per Event
# plt.figure(figsize=(10, 6))
# sns.histplot(event_hapax_counts['hapax_count'], bins=30, kde=True)
# plt.title('Histogram of Hapax Legomena Counts per Event')
# plt.xlabel('Hapax Legomena Count')
# plt.ylabel('Frequency')
# plt.show()

plt.figure(figsize=(12, 6))
sns.lineplot(data=event_hapax_counts, x='event', y='hapax_count', marker='o')
plt.title('Hapax Legomena Counts Across Events')
plt.xlabel('Event')
plt.ylabel('Hapax Legomena Count')
plt.xticks(rotation=45)
plt.show()


plt.figure(figsize=(15, 6))
sns.barplot(data=event_hapax_counts, x='event', y='hapax_count')
plt.title('Hapax Legomena Counts Across Events')
plt.xlabel('Event')
plt.ylabel('Hapax Legomena Count')
plt.xticks(rotation=45)
plt.show()

# Calculate proportions
proportions = event_hapax_counts.groupby('event_written')['event'].count() / event_hapax_counts['event'].nunique()
proportions = proportions.reset_index()
proportions.columns = ['Event Written', 'Proportion']

plt.figure(figsize=(8, 6))
sns.barplot(data=proportions, x='Event Written', y='Proportion')
plt.title('Proportion of Written vs. Non-Written Events with Hapaxes')
plt.xlabel('Event Written (1: Written, 0: Non-Written)')
plt.ylabel('Proportion of Events')
plt.show()


# Selecting and sorting the top events with the most hapaxes
top_events_count = 10
top_events = event_hapax_counts.sort_values('hapax_count', ascending=False).head(top_events_count)

# Creating a simple bar plot
plt.figure(figsize=(12, 6))
sns.barplot(data=top_events, x='event', y='hapax_count', hue='event_written', dodge=False)
plt.title(f'Top {top_events_count} Events with Most Hapax Legomena')
plt.xlabel('Event')
plt.ylabel('Hapax Legomena Count')
plt.legend(title='Event Written', labels=['Non-Written', 'Written'])
plt.show()


# Selecting a specific range of events for the scatter plot
selected_event_range = event_hapax_counts[event_hapax_counts['event'] <= 214]  # Selecting the first 200 events

plt.figure(figsize=(15, 6))
sns.regplot(data=selected_event_range, x='event', y='hapax_count', scatter=True, fit_reg=True, color='b', scatter_kws={'alpha':0.5})
plt.title('Scatter Plot of Hapax Legomena Counts with Trend Line')
plt.xlabel('Event')
plt.ylabel('Hapax Legomena Count')
plt.show()



# # Assuming your DataFrame has a 'chapter' column
# # Heatmap of Hapax Legomena Counts per Chapter
# # First, create a pivot table with chapter and event as indices and hapax count as values
# chapter_event_hapax = all_hapax_details_df.pivot_table(index='chapter', columns='event', values='hapax', aggfunc='count', fill_value=0)

# plt.figure(figsize=(12, 8))
# sns.heatmap(chapter_event_hapax, cmap='viridis', linewidths=.5)
# plt.title('Heatmap of Hapax Legomena Counts per Chapter and Event')
# plt.xlabel('Event')
# plt.ylabel('Chapter')
# plt.show()



##################### Statistical Tests #########################


# Count the number of hapaxes in each event
event_hapax_counts = all_hapax_details_df.groupby(['event', 'event_written']).size().reset_index(name='hapax_count')

# Separate the counts into written and non-written groups
written_counts = event_hapax_counts[event_hapax_counts['event_written'] == 1]['hapax_count']
non_written_counts = event_hapax_counts[event_hapax_counts['event_written'] == 0]['hapax_count']

# Mann-Whitney U Test
u_stat, mw_p_value = mannwhitneyu(written_counts, non_written_counts)

# Mean and Standard Deviation for written and non-written events
mean_written = np.mean(written_counts)
std_written = np.std(written_counts)
mean_non_written = np.mean(non_written_counts)
std_non_written = np.std(non_written_counts)




# Preparing the data for the Chi-Square Test
# Categorizing the counts into 'high' and 'low' based on the median value
median_hapax_count = event_hapax_counts['hapax_count'].median()
# event_hapax_counts['count_category'] = event_hapax_counts['hapax_count'].apply(lambda x: 'High' if x > median_hapax_count*2 else 'Low')

# Constructing the contingency table for the Chi-Square Test
chi2_table = pd.crosstab(event_hapax_counts['event_written'], event_hapax_counts['count_category'])

# Performing the Chi-Square Test
chi2_stat, chi2_p_value, chi2_dof, chi2_expected = chi2_contingency(chi2_table)

chi2_stat, chi2_p_value, chi2_dof, chi2_expected

# Categorizing hapax counts as 'High' (greater than 3 times median freq) or 'Low'
event_hapax_counts['count_category'] = event_hapax_counts['hapax_count'].apply(lambda x: 'High' if x > median_hapax_count*3 else 'Low')

# Creating the contingency table
chi2_table = pd.crosstab(event_hapax_counts['event_written'], event_hapax_counts['count_category'])

# Performing the Chi-Square Test
chi2_stat, chi2_p_value, chi2_dof, chi2_expected = chi2_contingency(chi2_table)

print("Chi-Squared Statistic:", chi2_stat)
print("P-value:", chi2_p_value)
print("Degrees of Freedom:", chi2_dof)
print("Expected Frequencies:", chi2_expected)







u_stat, mw_p_value, mean_written, std_written, mean_non_written, std_non_written, chi2_stat, chi2_p_value


print("Mann-Whitney U Test: U-statistic =", u_stat, ", P-value =", mw_p_value)
print("Mean (Written):", mean_written, ", Standard Deviation (Written):", std_written)
print("Mean (Non-Written):", mean_non_written, ", Standard Deviation (Non-Written):", std_non_written)

# Separate the hapax counts into two groups: written and non-written
written_hapax_counts = event_hapax_counts[event_hapax_counts['event_written'] == 1]['hapax_count']
non_written_hapax_counts = event_hapax_counts[event_hapax_counts['event_written'] == 0]['hapax_count']

# Perform a t-test
t_stat, p_value = ttest_ind(written_hapax_counts, non_written_hapax_counts, equal_var=False)

t_stat, p_value







