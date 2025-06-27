import oracledb
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime
from wordcloud import WordCloud # Will need to install this: pip install wordcloud

# --- Database Connection Details ---
DB_USER = "bank_reviews_user"
DB_PASSWORD = "bankreviewsuser" # REPLACE WITH YOUR ACTUAL PASSWORD
DB_HOST = "localhost"
DB_PORT = 1521
DB_SERVICE_NAME = "XEPDB1" # CONFIRM THIS MATCHES YOUR PDB (or XE)

# --- Output Directory for Visualizations ---
output_dir = 'visualizations'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"Created directory: {output_dir}")
else:
    print(f"Directory already exists: {output_dir}")

def load_data_from_oracle():
    connection = None
    try:
        print("Attempting to connect to Oracle database...")
        connection = oracledb.connect(
            user=DB_USER,
            password=DB_PASSWORD,
            host=DB_HOST,
            port=DB_PORT,
            service_name=DB_SERVICE_NAME
        )
        print("Successfully connected to Oracle database.")

        # Load Banks data
        banks_query = "SELECT bank_id, bank_name FROM Banks"
        banks_df = pd.read_sql(banks_query, connection)
        print(f"Loaded {len(banks_df)} banks.")
        # print("Columns in banks_df:", banks_df.columns.tolist()) # Debugging line, can keep or remove

        # Load Reviews data - IMPORTANT: Using TO_CHAR for CLOBs and correct column names
        reviews_query = """
        SELECT
            review_id, bank_id, TO_CHAR(review_text) AS review_text, rating, review_date, source,
            sentiment_label, sentiment_score, identified_themes
        FROM Reviews
        """
        reviews_df = pd.read_sql(reviews_query, connection)
        print(f"Loaded {len(reviews_df)} reviews.")
        # print("Columns in reviews_df:", reviews_df.columns.tolist()) # Debugging line, can keep or remove

        # Merge dataframes to include bank_name in reviews_df
        # Using 'BANK_ID' as confirmed by previous debugging
        reviews_df = pd.merge(reviews_df, banks_df, on='BANK_ID', how='left')
        print("Merged reviews with bank names.")

        return reviews_df

    except oracledb.Error as e:
        error_obj, = e.args
        print(f"Oracle Error Code: {error_obj.code}")
        print(f"Oracle Error Message: {error_obj.message}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None
    finally:
        if connection:
            connection.close()
            print("Database connection closed.")

if __name__ == "__main__":
    all_reviews_df = load_data_from_oracle()

    if all_reviews_df is not None:
        print("\nFirst 5 rows of loaded data:")
        print(all_reviews_df.head())
        print("\nData Types:")
        print(all_reviews_df.info())
        print("\nBasic Statistics for Ratings:")
        # Corrected column name to 'RATING'
        print(all_reviews_df['RATING'].describe())

        # --- INSIGHTS & RECOMMENDATIONS ---
        print("\n--- DERIVING INSIGHTS ---")

        # 1. Identify Drivers (Strengths) and Pain Points (Weaknesses)
        print("\n--- Drivers (Positive Themes) ---")
        # Ensure sentiment label is 'Positive' (case-sensitive)
        positive_themes = all_reviews_df[all_reviews_df['SENTIMENT_LABEL'] == 'Positive']['IDENTIFIED_THEMES'].value_counts()
        print("Top 5 Drivers:")
        print(positive_themes.head(5))

        print("\n--- Pain Points (Negative Themes) ---")
        # Ensure sentiment label is 'Negative' (case-sensitive)
        negative_themes = all_reviews_df[all_reviews_df['SENTIMENT_LABEL'] == 'Negative']['IDENTIFIED_THEMES'].value_counts()
        print("Top 5 Pain Points:")
        print(negative_themes.head(5))

        print("\n--- Average Rating by Identified Theme ---")
        # Corrected column name to 'RATING'
        avg_rating_by_theme = all_reviews_df.groupby('IDENTIFIED_THEMES')['RATING'].mean().sort_values(ascending=False)
        print("Themes with highest average ratings:")
        print(avg_rating_by_theme.head(10))
        print("\nThemes with lowest average ratings:")
        print(avg_rating_by_theme.tail(10))


        # 2. Compare Banks
        print("\n--- BANK COMPARISONS ---")
        # Corrected column name to 'RATING'
        print("\nAverage Rating by Bank:")
        avg_rating_by_bank = all_reviews_df.groupby('BANK_NAME')['RATING'].mean().sort_values(ascending=False)
        print(avg_rating_by_bank)

        print("\nSentiment Distribution by Bank:")
        # Ensure 'SENTIMENT_LABEL' and 'BANK_NAME' are correctly cased
        sentiment_by_bank = all_reviews_df.groupby('BANK_NAME')['SENTIMENT_LABEL'].value_counts(normalize=True).unstack().fillna(0)
        print(sentiment_by_bank)

        print("\nTop 3 Pain Points per Bank:")
        for bank_name in all_reviews_df['BANK_NAME'].unique():
            print(f"\n--- {bank_name} ---")
            bank_negative_themes = all_reviews_df[
                (all_reviews_df['BANK_NAME'] == bank_name) &
                (all_reviews_df['SENTIMENT_LABEL'] == 'Negative')
            ]['IDENTIFIED_THEMES'].value_counts().head(3)
            if not bank_negative_themes.empty:
                print(bank_negative_themes)
            else:
                print("No significant negative themes identified for this bank.")

        # --- Recommendations based on Insights (Manually formulated based on output) ---
        print("\n--- RECOMMENDATIONS ---")
        print("Based on the above insights, here are some recommendations:")
        print("1. Prioritize 'Crashes/Bugs' and 'Transaction Performance' fixes: If these are top pain points, immediate attention is needed for app stability and speed.")
        print("2. Enhance 'Customer Support' for specific banks: If one bank consistently has lower sentiment/ratings for support, target training or improved in-app help for them.")
        print("3. Leverage 'Ease of Use' as a key feature: If 'Ease of Use' is a strong driver for positive sentiment, promote this feature in marketing and ensure new features maintain simplicity.")
        print("4. Investigate 'Other' category reviews: The 'Other' category in themes can be a catch-all. Reviewing these texts manually might reveal new, uncategorized insights for further app improvement.")


        # --- VISUALIZATIONS ---
        print("\n--- GENERATING VISUALIZATIONS ---")

        # 1. Overall Rating Distribution
        plt.figure(figsize=(8, 5))
        sns.countplot(data=all_reviews_df, x='RATING', palette='viridis')
        plt.title('Overall Rating Distribution')
        plt.xlabel('Rating (1-5)')
        plt.ylabel('Number of Reviews')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'overall_rating_distribution.png'))
        # plt.show() # Uncomment if you want plots to pop up immediately

        # 2. Overall Sentiment Distribution
        plt.figure(figsize=(8, 5))
        sns.countplot(data=all_reviews_df, x='SENTIMENT_LABEL', order=all_reviews_df['SENTIMENT_LABEL'].value_counts().index, palette='coolwarm')
        plt.title('Overall Sentiment Distribution')
        plt.xlabel('Sentiment Label')
        plt.ylabel('Number of Reviews')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'overall_sentiment_distribution.png'))
        # plt.show()

        # 3. Sentiment Distribution by Bank
        plt.figure(figsize=(12, 6))
        sns.countplot(data=all_reviews_df, x='BANK_NAME', hue='SENTIMENT_LABEL', palette='muted')
        plt.title('Sentiment Distribution by Bank')
        plt.xlabel('Bank Name')
        plt.ylabel('Number of Reviews')
        plt.xticks(rotation=45, ha='right')
        plt.legend(title='Sentiment')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'sentiment_by_bank.png'))
        # plt.show()

        # 4. Top 10 Identified Themes
        # Adjust 'Other' handling if it's too dominant and uninformative for your data
        plt.figure(figsize=(10, 6))
        # Filter out 'Other' if it's not insightful, or adjust the head() number
        themes_for_plot = all_reviews_df['IDENTIFIED_THEMES'].value_counts()
        if 'Other' in themes_for_plot.index:
            themes_for_plot = themes_for_plot.drop('Other')
        top_themes = themes_for_plot.head(10) # Get top 10 after potential 'Other' removal
        sns.barplot(x=top_themes.index, y=top_themes.values, palette='plasma')
        plt.title('Top Identified Themes in Reviews (Excluding "Other" if present)')
        plt.xlabel('Theme')
        plt.ylabel('Number of Occurrences')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'top_themes.png'))
        # plt.show()

        # 5. Rating vs Sentiment Score Scatter Plot
        plt.figure(figsize=(10, 6))
        # Ensure correct column names: 'RATING', 'SENTIMENT_SCORE', 'SENTIMENT_LABEL'
        sns.scatterplot(data=all_reviews_df, x='RATING', y='SENTIMENT_SCORE', hue='SENTIMENT_LABEL', palette='viridis', alpha=0.7)
        plt.title('Rating vs Sentiment Score')
        plt.xlabel('Rating')
        plt.ylabel('Sentiment Score')
        plt.grid(linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'rating_vs_sentiment_score.png'))
        # plt.show()

        # Optional: Word Cloud (Requires pip install wordcloud)
        try:
            print("\nGenerating Word Cloud (requires 'wordcloud' library)...")
            all_text = ' '.join(all_reviews_df['REVIEW_TEXT'].dropna().astype(str)) # Ensure text is string
            wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_text)
            plt.figure(figsize=(10, 5))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.title('Word Cloud of Review Text')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'review_wordcloud.png'))
            # plt.show()
            print("Word Cloud generated successfully.")
        except ImportError:
            print("Skipping Word Cloud: 'wordcloud' library not installed. Run 'pip install wordcloud' to enable.")
        except Exception as e:
            print(f"Error generating Word Cloud: {e}")

        print("\nAll visualizations generated and saved in the 'visualizations/' folder.")

    else:
        print("Failed to load data from Oracle. Cannot proceed with insights and visualizations.")