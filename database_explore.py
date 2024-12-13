import sqlite3
import pandas as pd
import os

# Input database name on terminal
database_name = input("Enter the database name: ")

# Connect to the database
conn = sqlite3.connect(database_name)
cursor = conn.cursor()

# Show all tables and choose a table
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = cursor.fetchall()
print("\nTables:")
for table in tables:
    print(f"- {table[0]}")  # Print each table name on a new line
table_name = input("Enter the table name: ")

# Input what you want to do
action = input("What do you want to do? \n"
               "1. Add a new column \n"
               "2. Delete a column \n"
               "3. Show all columns in a table \n"
               "4. Convert table to CSV \n"
               "5. Exit \n")

if action == '1':
    new_column_name = input("Enter the name of the new column: ")
    cursor.execute(f"ALTER TABLE {table_name} ADD COLUMN {new_column_name} TEXT;")
    conn.commit()
    print(f"Column '{new_column_name}' added successfully.")

elif action == '2':
    # Show all columns in the table
    cursor.execute(f"PRAGMA table_info({table_name});")
    columns = cursor.fetchall()
    print(f"\nColumns in {table_name}:")
    for column in columns:
        print(f"- {column[1]} ({column[2]})")  # Print each column name and type
    column_to_delete = input("Enter the name of the column to delete: ")
    
    # SQLite does not support dropping columns directly, so we need to create a new table
    cursor.execute(f"PRAGMA table_info({table_name});")
    columns = cursor.fetchall()
    column_names = [column[1] for column in columns if column[1] != column_to_delete]
    
    # Create a new table with the remaining columns
    new_table_name = f"{table_name}_modified"
    cursor.execute(f"CREATE TABLE {new_table_name} ({', '.join(column_names)});")
    
    # Copy data from the old table to the new table
    cursor.execute(f"INSERT INTO {new_table_name} ({', '.join(column_names)}) SELECT {', '.join(column_names)} FROM {table_name};")
    
    conn.commit()
    print(f"Column '{column_to_delete}' deleted successfully. The modified table is stored as '{new_table_name}'.")
    # stored information in a new file
    with open(f"{new_table_name}.sql", "w") as file:
        file.write(f"CREATE TABLE {new_table_name} ({', '.join(column_names)});\n")
        file.write(f"INSERT INTO {new_table_name} ({', '.join(column_names)}) SELECT {', '.join(column_names)} FROM {table_name};\n")
    print(f"Information stored in {new_table_name}.sql")

elif action == '3':
    # Show all columns in the selected table
    cursor.execute(f"PRAGMA table_info({table_name});")
    columns = cursor.fetchall()
    print(f"\nColumns in {table_name}:")
    for column in columns:
        print(f"- {column[1]} ({column[2]})")  # Print each column name and type

elif action == '4':
    # Convert the selected table to CSV
    df = pd.read_sql_query(f"SELECT * FROM {table_name};", conn)
    csv_file_name = f"{table_name}.csv"
    df.to_csv(csv_file_name, index=False)
    print(f"Table '{table_name}' has been converted to CSV and saved as '{csv_file_name}' in the current directory.")

elif action == '5':
    print("Exiting the program.")

else:
    print("Invalid action selected.")

# Close the connection
conn.close()
