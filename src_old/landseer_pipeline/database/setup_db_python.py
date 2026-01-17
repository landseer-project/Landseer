#!/usr/bin/env python3
"""
Python-based database setup script for Landseer Pipeline.

Use this if you don't have the mysql CLI client installed.

Usage:
    python setup_db_python.py --create-db --create-user --password landseer
"""

import argparse
import os
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Setup Landseer MySQL database")
    parser.add_argument("--host", default=os.getenv("LANDSEER_DB_HOST", "localhost"))
    parser.add_argument("--port", type=int, default=int(os.getenv("LANDSEER_DB_PORT", "3306")))
    parser.add_argument("--database", default=os.getenv("LANDSEER_DB_NAME", "landseer_pipeline"))
    parser.add_argument("--user", default=os.getenv("LANDSEER_DB_USER", "landseer"))
    parser.add_argument("--password", default=os.getenv("LANDSEER_DB_PASSWORD", ""))
    parser.add_argument("--root-user", default="root")
    parser.add_argument("--root-password", default="")
    parser.add_argument("--create-db", action="store_true", help="Create database")
    parser.add_argument("--create-user", action="store_true", help="Create user")
    parser.add_argument("--schema-only", action="store_true", help="Only apply schema")
    
    args = parser.parse_args()
    
    try:
        import mysql.connector
    except ImportError:
        print("ERROR: mysql-connector-python not installed.")
        print("Install it with: pip install mysql-connector-python")
        sys.exit(1)
    
    schema_file = Path(__file__).parent / "schema.sql"
    if not schema_file.exists():
        print(f"ERROR: Schema file not found: {schema_file}")
        sys.exit(1)
    
    # Connect as root for admin operations
    if args.create_db or args.create_user:
        print(f"Connecting to MySQL as {args.root_user}...")
        try:
            conn = mysql.connector.connect(
                host=args.host,
                port=args.port,
                user=args.root_user,
                password=args.root_password
            )
            cursor = conn.cursor()
            
            if args.create_db:
                print(f"Creating database: {args.database}")
                cursor.execute(f"""
                    CREATE DATABASE IF NOT EXISTS {args.database}
                    CHARACTER SET utf8mb4
                    COLLATE utf8mb4_unicode_ci
                """)
                print(f"  ✓ Database '{args.database}' created")
            
            if args.create_user:
                print(f"Creating user: {args.user}")
                try:
                    cursor.execute(f"CREATE USER IF NOT EXISTS '{args.user}'@'%' IDENTIFIED BY '{args.password}'")
                except mysql.connector.Error:
                    # User might already exist
                    pass
                cursor.execute(f"GRANT ALL PRIVILEGES ON {args.database}.* TO '{args.user}'@'%'")
                cursor.execute("FLUSH PRIVILEGES")
                print(f"  ✓ User '{args.user}' created with full access to '{args.database}'")
            
            conn.commit()
            cursor.close()
            conn.close()
            
        except mysql.connector.Error as e:
            print(f"ERROR: Failed to connect as root: {e}")
            print("Make sure MySQL is running and root credentials are correct.")
            sys.exit(1)
    
    # Apply schema
    print(f"\nApplying schema to {args.database}...")
    try:
        conn = mysql.connector.connect(
            host=args.host,
            port=args.port,
            user=args.user if not args.create_user else args.root_user,
            password=args.password if not args.create_user else args.root_password,
            database=args.database
        )
        cursor = conn.cursor()
        
        # Read and execute schema
        schema_sql = schema_file.read_text()
        
        # Split by statements, handling DELIMITER changes
        statements = []
        current_delimiter = ";"
        current_statement = []
        
        for line in schema_sql.split('\n'):
            stripped = line.strip()
            
            # Handle DELIMITER changes
            if stripped.upper().startswith("DELIMITER"):
                parts = stripped.split()
                if len(parts) >= 2:
                    if current_statement:
                        stmt = '\n'.join(current_statement).strip()
                        if stmt:
                            statements.append(stmt)
                        current_statement = []
                    current_delimiter = parts[1]
                continue
            
            current_statement.append(line)
            
            if stripped.endswith(current_delimiter):
                stmt = '\n'.join(current_statement)
                stmt = stmt[:stmt.rfind(current_delimiter)].strip()
                if stmt:
                    statements.append(stmt)
                current_statement = []
        
        if current_statement:
            stmt = '\n'.join(current_statement).strip()
            if stmt:
                statements.append(stmt)
        
        # Execute statements
        success = 0
        skipped = 0
        for stmt in statements:
            if not stmt.strip():
                continue
            try:
                cursor.execute(stmt)
                conn.commit()
                success += 1
            except mysql.connector.Error as e:
                if "already exists" in str(e).lower():
                    skipped += 1
                else:
                    print(f"  Warning: {e}")
        
        print(f"  ✓ Schema applied: {success} statements executed, {skipped} skipped (already exist)")
        
        cursor.close()
        conn.close()
        
    except mysql.connector.Error as e:
        print(f"ERROR: Failed to apply schema: {e}")
        sys.exit(1)
    
    print("\n" + "="*60)
    print("DATABASE SETUP COMPLETE")
    print("="*60)
    print(f"Host:     {args.host}")
    print(f"Port:     {args.port}")
    print(f"Database: {args.database}")
    print(f"User:     {args.user}")
    print()
    print("To use with Landseer, set these environment variables:")
    print(f"  export LANDSEER_DB_HOST={args.host}")
    print(f"  export LANDSEER_DB_PORT={args.port}")
    print(f"  export LANDSEER_DB_NAME={args.database}")
    print(f"  export LANDSEER_DB_USER={args.user}")
    print(f"  export LANDSEER_DB_PASSWORD=<your_password>")


if __name__ == "__main__":
    main()
