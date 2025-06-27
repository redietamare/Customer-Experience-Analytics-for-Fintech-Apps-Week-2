from google_play_scraper import Sort, reviews
import pandas as pd
import datetime

# Define the app IDs for each bank
# IMPORTANT: Replace these with the actual app IDs you find on the Play Store
app_ids = {
    "Commercial Bank of Ethiopia": "com.combanketh.mobilebanking", # Example ID, find actual
    "Bank of Abyssinia": "com.boa.boaMobileBanking",       # Example ID, find actual
    "Dashen Bank": "com.dashen.dashensuperapppyr"       # Example ID, find actual
}

all_reviews = []
min_reviews_per_bank = 400

print("Starting review scraping...")

for bank_name, app_id in app_ids.items():
    print(f"\nScraping reviews for {bank_name} (App ID: {app_id})...")
    try:
        # You can adjust count to get more or fewer reviews.
        # Use 'sort=Sort.NEWEST' to get the most recent reviews.
        # Using 'count' directly can be tricky for exact numbers,
        # as it might return less if there aren't enough or more if it hits a page limit.
        # We'll aim for 400+ and then handle duplicates/missing later.
        result, continuation_token = reviews(
            app_id,
            lang='en',      # Language of reviews
            country='et',   # Country (Ethiopia)
            sort=Sort.NEWEST, # Get newest reviews first
            count=min_reviews_per_bank + 100 # Request a bit more to ensure we get enough after cleaning
        )

        # Process scraped reviews
        for r in result:
            all_reviews.append({
                'review_text': r['content'],
                'rating': r['score'],
                'date': r['at'],
                'bank_name': bank_name,
                'source': 'Google Play Store'
            })
        print(f"Successfully scraped {len(result)} reviews for {bank_name}.")

    except Exception as e:
        print(f"Error scraping reviews for {bank_name}: {e}")
        print("Please ensure the App ID is correct and the app exists in the specified country/language.")

# Convert to DataFrame
df = pd.DataFrame(all_reviews)

print(f"\nTotal reviews scraped before preprocessing: {len(df)}")

# --- Preprocessing ---
print("\nStarting preprocessing...")

# 1. Handle Duplicates
# Drop rows where 'review_text' and 'bank_name' are identical, keeping the first occurrence.
# We include 'bank_name' in the subset to avoid dropping reviews that might be identical
# in text but refer to different banks (though unlikely for app reviews).
initial_rows = len(df)
df.drop_duplicates(subset=['review_text', 'bank_name'], inplace=True)
print(f"Removed {initial_rows - len(df)} duplicate reviews.")

# 2. Handle Missing Data
# Check for missing values
print("Missing values before handling:")
print(df.isnull().sum())
# Drop rows where 'review_text' or 'rating' is missing, as these are critical for analysis
df.dropna(subset=['review_text', 'rating'], inplace=True)
print("Missing values after handling (dropped rows with missing review_text or rating):")
print(df.isnull().sum())


# 3. Normalize Dates
# Convert 'date' column to datetime objects, then format to YYYY-MM-DD
df['date'] = pd.to_datetime(df['date'])
df['date'] = df['date'].dt.strftime('%Y-%m-%d')
print("Dates normalized to YYYY-MM-DD format.")

# Rename columns for clarity (as specified in the assignment)
df = df.rename(columns={'review_text': 'review', 'bank_name': 'bank'})

# Ensure minimum reviews per bank
for bank_name in app_ids.keys():
    bank_df = df[df['bank'] == bank_name]
    if len(bank_df) < min_reviews_per_bank:
        print(f"WARNING: Only {len(bank_df)} reviews collected for {bank_name}. Target was {min_reviews_per_bank}.")
    else:
        print(f"Collected {len(bank_df)} reviews for {bank_name} (target met).")

print(f"\nTotal unique and cleaned reviews: {len(df)}")
print("First 5 rows of the cleaned data:")
print(df.head())
print("\nData types:")
print(df.info())

# Save to CSV
output_filename = 'bank_app_reviews.csv'
df.to_csv(output_filename, index=False, encoding='utf-8')
print(f"\nCleaned data saved to {output_filename}")