# Entity Resolution for NPI Doctor and Patentee Matching
---

## Project Overview

This project implements an **entity resolution pipeline** to match doctor records from a National Provider Identifier (NPI) dataset to organization (patentee) records from a separate dataset, **without shared unique IDs**.  
The main challenge: names often appeared inconsistently, with variations due to typos, abbreviations, and formatting differences.

The goal was to **link the same entities** across datasets despite these imperfections.

---
## Problem Context

In real-world datasets, especially in healthcare, finance, and government systems, perfect identifiers are often missing. Instead, messy and inconsistent data must be linked by resolving entities that **appear differently but refer to the same identity**.

**Challenges addressed:**
- No shared IDs between doctors and patentees
- Frequent spelling variations, typos, and abbreviations
- Need for a scalable solution to match thousands of records

---

## Approach

### 1. Data Augmentation via Simulated Typos
- Generated synthetic training data by introducing random character swaps, deletions, and substitutions into real names
- Created "positive" and "negative" pairs to train the model
- Example: Sara Kelemen → Sra Keleemn

### 2. Feature Engineering
- Computed **Jaro-Winkler similarity** and **Levenshtein distance** for first and last names separately
- Created numeric feature vectors representing pairwise name similarities

### 3. Supervised Learning Model
- Trained a **XGBoost model** on the similarity scores from a reusable classifier
- Model learned which types of discrepancies are typical in true matches

### 4. Blocking Strategy for Scalability
- Used **blocking by first letter of the last name** to reduce the number of comparisons
- Balanced efficiency and recall

### 5. Results and Storage
- Matched pairs with high probability stored in a new **SQLite bridge table**
- Created an efficient mapping between doctor and patentee datasets

---
## Final: Why Entity Resolution Matters
- Accurate entity resolution ensures that systems connect the correct people, organizations, and records.
- In sectors like healthcare, finance, and e-commerce, trustworthy linking of messy data is vital for analytics, decision-making, and operational success.
---
## Technical Stack

Check pyproject.toml for repository dependencies. 

---

