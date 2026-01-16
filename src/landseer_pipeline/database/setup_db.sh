#!/bin/bash
# =====================================================
# Landseer Pipeline Database Setup Script
# =====================================================
# This script sets up the MySQL database for storing
# pipeline run results.
# =====================================================

set -e

# Default configuration
DB_HOST="${LANDSEER_DB_HOST:-localhost}"
DB_PORT="${LANDSEER_DB_PORT:-3306}"
DB_NAME="${LANDSEER_DB_NAME:-landseer_pipeline}"
DB_USER="${LANDSEER_DB_USER:-landseer}"
DB_PASSWORD="${LANDSEER_DB_PASSWORD:-}"
MYSQL_ROOT_PASSWORD="${MYSQL_ROOT_PASSWORD:-}"

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCHEMA_FILE="${SCRIPT_DIR}/schema.sql"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --host HOST           MySQL host (default: localhost)"
    echo "  --port PORT           MySQL port (default: 3306)"
    echo "  --database NAME       Database name (default: landseer_pipeline)"
    echo "  --user USER           Database user (default: landseer)"
    echo "  --password PASS       Database password"
    echo "  --root-password PASS  MySQL root password (for creating user/database)"
    echo "  --create-user         Create database user (requires root password)"
    echo "  --create-db           Create database (requires root password)"
    echo "  --schema-only         Only apply schema (database must exist)"
    echo "  --help                Show this help message"
    echo ""
    echo "Environment variables:"
    echo "  LANDSEER_DB_HOST      MySQL host"
    echo "  LANDSEER_DB_PORT      MySQL port"
    echo "  LANDSEER_DB_NAME      Database name"
    echo "  LANDSEER_DB_USER      Database user"
    echo "  LANDSEER_DB_PASSWORD  Database password"
    echo "  MYSQL_ROOT_PASSWORD   MySQL root password"
}

# Parse command line arguments
CREATE_USER=false
CREATE_DB=false
SCHEMA_ONLY=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --host)
            DB_HOST="$2"
            shift 2
            ;;
        --port)
            DB_PORT="$2"
            shift 2
            ;;
        --database)
            DB_NAME="$2"
            shift 2
            ;;
        --user)
            DB_USER="$2"
            shift 2
            ;;
        --password)
            DB_PASSWORD="$2"
            shift 2
            ;;
        --root-password)
            MYSQL_ROOT_PASSWORD="$2"
            shift 2
            ;;
        --create-user)
            CREATE_USER=true
            shift
            ;;
        --create-db)
            CREATE_DB=true
            shift
            ;;
        --schema-only)
            SCHEMA_ONLY=true
            shift
            ;;
        --help)
            usage
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# Check for required tools
if ! command -v mysql &> /dev/null; then
    print_error "MySQL client not found. Please install mysql-client."
    exit 1
fi

# Check for schema file
if [[ ! -f "$SCHEMA_FILE" ]]; then
    print_error "Schema file not found: $SCHEMA_FILE"
    exit 1
fi

# Function to run MySQL command as root
mysql_root() {
    if [[ -n "$MYSQL_ROOT_PASSWORD" ]]; then
        mysql -h "$DB_HOST" -P "$DB_PORT" -u root -p"$MYSQL_ROOT_PASSWORD" "$@"
    else
        mysql -h "$DB_HOST" -P "$DB_PORT" -u root "$@"
    fi
}

# Function to run MySQL command as user
mysql_user() {
    if [[ -n "$DB_PASSWORD" ]]; then
        mysql -h "$DB_HOST" -P "$DB_PORT" -u "$DB_USER" -p"$DB_PASSWORD" "$@"
    else
        mysql -h "$DB_HOST" -P "$DB_PORT" -u "$DB_USER" "$@"
    fi
}

# Create user if requested
if [[ "$CREATE_USER" == true ]]; then
    print_info "Creating database user: $DB_USER"
    
    if [[ -z "$MYSQL_ROOT_PASSWORD" ]] && [[ "$DB_HOST" != "localhost" ]]; then
        print_warn "No root password provided for remote host"
    fi
    
    mysql_root -e "
        CREATE USER IF NOT EXISTS '$DB_USER'@'%' IDENTIFIED BY '$DB_PASSWORD';
        GRANT ALL PRIVILEGES ON $DB_NAME.* TO '$DB_USER'@'%';
        FLUSH PRIVILEGES;
    "
    
    print_info "User '$DB_USER' created successfully"
fi

# Create database if requested
if [[ "$CREATE_DB" == true ]]; then
    print_info "Creating database: $DB_NAME"
    
    mysql_root -e "
        CREATE DATABASE IF NOT EXISTS $DB_NAME
        CHARACTER SET utf8mb4
        COLLATE utf8mb4_unicode_ci;
    "
    
    print_info "Database '$DB_NAME' created successfully"
fi

# Apply schema
if [[ "$SCHEMA_ONLY" == true ]] || [[ "$CREATE_DB" == true ]] || [[ "$CREATE_USER" == false && "$CREATE_DB" == false ]]; then
    print_info "Applying database schema..."
    
    # Try with user credentials first, fall back to root
    if mysql_user "$DB_NAME" < "$SCHEMA_FILE" 2>/dev/null; then
        print_info "Schema applied successfully using user credentials"
    elif mysql_root "$DB_NAME" < "$SCHEMA_FILE" 2>/dev/null; then
        print_info "Schema applied successfully using root credentials"
    else
        print_error "Failed to apply schema. Check database credentials."
        exit 1
    fi
fi

print_info "Database setup complete!"
print_info ""
print_info "Connection details:"
print_info "  Host:     $DB_HOST"
print_info "  Port:     $DB_PORT"
print_info "  Database: $DB_NAME"
print_info "  User:     $DB_USER"
print_info ""
print_info "To connect with Python, use:"
print_info "  DATABASE_URL=mysql+mysqlconnector://$DB_USER:***@$DB_HOST:$DB_PORT/$DB_NAME"
