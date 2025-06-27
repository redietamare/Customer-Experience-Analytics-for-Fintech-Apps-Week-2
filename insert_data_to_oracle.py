import oracledb
import pandas as pd
import os
from datetime import datetime

# --- 1. Database Connection Details ---
DB_USER = "bank_reviews_user"
DB_PASSWORD = "bankreviewsuser" # REPLACE WITH YOUR ACTUAL PASSWORD
DB_HOST = "localhost"
DB_PORT = 1521
DB_SERVICE_NAME = "XEPDB1" # CONFIRM THIS MATCHES YOUR PDB (or XE)

# --- 2. Path to your Cleaned Data ---
CLEANED_DATA_PATH = 'bank_app_reviews_analyzed.csv'

# --- 3. Main Data Insertion Logic ---
def insert_data_to_oracle():
    connection = None
    cursor = None
    try:
        # Establish database connection
        print("Attempting to connect to Oracle database...")
        connection = oracledb.connect(
            user=DB_USER,
            password=DB_PASSWORD,
            host=DB_HOST,
            port=DB_PORT,
            service_name=DB_SERVICE_NAME
        )
        cursor = connection.cursor()
        print("Successfully connected to Oracle database.")

        # --- Read Cleaned Data ---
        print(f"Reading cleaned data from: {CLEANED_DATA_PATH}")
        try:
            df = pd.read_csv(CLEANED_DATA_PATH)
            print(f"Loaded {len(df)} rows from cleaned data.")
            print("Columns in loaded DataFrame:", df.columns.tolist())
        except FileNotFoundError:
            print(f"Error: Cleaned data file not found at {CLEANED_DATA_PATH}")
            return
        except Exception as e:
            print(f"Error loading data: {e}")
            return

        # --- Insert into Banks Table ---
        print("Inserting unique bank names into Banks table...")
        unique_banks = df['bank'].dropna().unique() # This one is already correct!
        bank_id_map = {}

        banks_to_insert = []
        for bank_name in unique_banks:
            banks_to_insert.append({"bank_name": bank_name})

        for bank_name in unique_banks:
            cursor.execute("SELECT bank_id FROM Banks WHERE bank_name = :bank_name", bank_name=bank_name)
            existing_bank_id = cursor.fetchone()
            if existing_bank_id:
                bank_id_map[bank_name] = existing_bank_id[0]
            else:
                id_var = cursor.var(oracledb.NUMBER)
                cursor.execute(
                    "INSERT INTO Banks (bank_name) VALUES (:bank_name) RETURNING bank_id INTO :bank_id",
                    bank_name=bank_name,
                    bank_id=id_var
                )
                new_bank_id = id_var.getvalue()[0]
                bank_id_map[bank_name] = new_bank_id
        print(f"Finished inserting {len(bank_id_map)} unique banks.")


        # --- Insert into Reviews Table ---
        print("Preparing reviews data for insertion...")
        reviews_data_for_db = []
        for index, row in df.iterrows():
            # CHANGE 1: Access 'bank' column, not 'Bank Name'
            bank_id = bank_id_map.get(row['bank'])
            if bank_id is None:
                # CHANGE 2: Update warning message to use 'bank'
                print(f"Warning: Could not find bank_id for '{row['bank']}'. Skipping review.")
                continue

            reviews_data_for_db.append((
                bank_id,
                # CHANGE 3: Use 'review'
                str(row['review']),
                # CHANGE 4: Use 'rating'
                int(row['rating']),
                # CHANGE 5: Use 'date' and ensure format matches your CSV
                datetime.strptime(str(row['date']), '%Y-%m-%d'),
                # CHANGE 6: Use 'source'
                str(row['source']),
                # CHANGE 7: Use 'sentiment_label'
                row.get('sentiment_label'),
                # CHANGE 8: Use 'sentiment_score'
                row.get('sentiment_score'),
                # CHANGE 9: Use 'identified_themes'
                row.get('identified_themes')
            ))

        print(f"Inserting {len(reviews_data_for_db)} reviews into Reviews table...")
        cursor.executemany(
            """INSERT INTO Reviews (
                bank_id, review_text, rating, review_date, source,
                sentiment_label, sentiment_score, identified_themes
            ) VALUES (
                :1, :2, :3, :4, :5, :6, :7, :8
            )""",
            reviews_data_for_db
        )
        print(f"Successfully inserted {cursor.rowcount} reviews.")

        connection.commit()
        print("Data insertion committed successfully.")

    except oracledb.Error as e:
        error_obj, = e.args
        print(f"Oracle Error Code: {error_obj.code}")
        print(f"Oracle Error Message: {error_obj.message}")
        print("Transaction rolled back due to error.")
        if connection:
            connection.rollback()
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        print("Transaction rolled back due to error.")
        if connection:
            connection.rollback()
    finally:
        if cursor:
            cursor.close()
            print("Cursor closed.")
        if connection:
            connection.close()
            print("Database connection closed.")

if __name__ == "__main__":
    insert_data_to_oracle()