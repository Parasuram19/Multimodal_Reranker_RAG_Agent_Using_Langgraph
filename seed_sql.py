"""
seed_sql.py - Fixed version with robust INSERT logic
Generate and insert seed data for:
- reliance_shareholders
- jio_users  
- jio_bp_users
"""

import json
import random
import uuid
from datetime import datetime, timedelta, date
from decimal import Decimal
from typing import List, Dict, Any, Optional

import psycopg2
from psycopg2.extras import execute_batch, RealDictCursor
from dotenv import load_dotenv
import os
import traceback

load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATABASE_URL = os.getenv("ADMIN_DB_URL", "postgresql://user:pass@localhost:5432/reliance_knowledge")
BATCH_SIZE = 200  # Reduced for better stability

# ---------------------------------------------------------------------------
# Sample Data Pools
# ---------------------------------------------------------------------------
REGIONS = ["North", "South", "East", "West", "Central", "Metro", "Tier-2", "Tier-3"]
SUBSCRIPTION_PLANS = {
    "jio": ["JioPrime", "JioPostpaid Plus", "JioPrepaid", "JioFiber", "JioAirFiber"],
    "jio_bp": ["BP Retail", "BP Fleet", "BP Enterprise", "BP Partner"],
    "reliance": ["Retail Investor", "Institutional", "Promoter", "FII/DII", "Employee ESOP"]
}
SHARE_CATEGORIES = ["Equity", "Preference", "Convertible Debenture", "Warrants", "ESOP"]
CIRCLES = ["Mumbai", "Delhi", "Kolkata", "Chennai", "Bangalore", "Hyderabad", "Pune", "Ahmedabad"]
STATES = ["Maharashtra", "Gujarat", "Karnataka", "Tamil Nadu", "Delhi", "UP", "Rajasthan", "WB", "MP"]

# ---------------------------------------------------------------------------
# Seed Data Generators (with better None handling)
# ---------------------------------------------------------------------------

def _safe_decimal(value: Any, default: Optional[Decimal] = None) -> Optional[Decimal]:
    """Safely convert to Decimal or return None."""
    if value is None:
        return default
    try:
        return Decimal(str(value))
    except:
        return default

def generate_reliance_shareholders(n: int = 1000) -> List[Dict[str, Any]]:
    """Generate seed data for reliance_shareholders."""
    users = []
    base_date = datetime(2023, 1, 1)
    
    for i in range(n):
        user_id = str(uuid.uuid4())
        holder_type = random.choice(SUBSCRIPTION_PLANS["reliance"])
        is_retail = holder_type == "Retail Investor"
        shares_held = random.randint(1, 10000) if is_retail else random.randint(10000, 5000000)
        price = random.uniform(2200, 2800)
        
        users.append({
            "user_id": user_id,
            "holder_name": f"Shareholder_{i+1:05d}",
            "holder_type": holder_type,
            "pan_number": f"ABCDE{i+1:04d}F" if is_retail else None,
            "demat_account": f"IN30{random.randint(1000,9999)}/{random.randint(100000,999999)}",
            "shares_held": shares_held,
            "share_category": random.choice(SHARE_CATEGORIES),
            "avg_buy_price": round(price, 2),
            "current_value_in_cr": round(shares_held * random.uniform(2400, 2700) / 10000000, 4),
            "percentage_holding": round(random.uniform(0.001, 5.0) if holder_type != "Promoter" else random.uniform(50, 52), 4),
            "first_purchase_date": (base_date + timedelta(days=random.randint(0, 700))).date(),
            "last_transaction_date": (datetime.now() - timedelta(days=random.randint(0, 30))).date(),
            "voting_rights": shares_held,
            "region": random.choice(REGIONS),
            "is_active": random.random() > 0.1,
            "kyc_status": random.choice(["Verified", "Pending", "Expired"]),
            "created_at": datetime.now(),
            "updated_at": datetime.now(),
        })
    return users


def generate_jio_users(n: int = 5000) -> List[Dict[str, Any]]:
    """Generate seed data for jio_users."""
    users = []
    base_date = datetime(2022, 6, 1)
    
    for i in range(n):
        user_id = str(uuid.uuid4())
        plan = random.choice(SUBSCRIPTION_PLANS["jio"])
        registration = base_date + timedelta(days=random.randint(0, 900))
        
        users.append({
            "user_id": user_id,
            "mobile_number": f"+91{random.randint(7000000000, 9999999999)}",
            "email": f"jio.user{i+1:05d}@example.com",
            "subscription_plan": plan,
            "plan_category": random.choice(["Prepaid", "Postpaid", "Fiber", "Enterprise"]),
            "registration_date": registration.date(),
            "last_activity_timestamp": datetime.now() - timedelta(hours=random.randint(1, 720)),
            "is_active": random.random() > 0.15,
            "region": random.choice(REGIONS),
            "circle": random.choice(CIRCLES),
            "data_usage_gb_30d": round(random.uniform(1, 500), 2),
            "voice_minutes_30d": random.randint(0, 5000),
            "sms_count_30d": random.randint(0, 1000),
            "arpu_inr": round(random.uniform(150, 2500), 2),
            "recharge_frequency_days": random.randint(7, 90),
            "linked_jfs_account": random.random() > 0.7,
            "jfs_shares_held": random.randint(0, 10000) if random.random() > 0.7 else 0,
            "loyalty_points": random.randint(0, 50000),
            "churn_risk_score": round(random.uniform(0, 1), 3),
            "nps_score": random.randint(-100, 100),
            "created_at": datetime.now(),
            "updated_at": datetime.now(),
        })
    return users


def generate_jio_bp_users(n: int = 2000) -> List[Dict[str, Any]]:
    """Generate seed data for jio_bp_users."""
    users = []
    base_date = datetime(2023, 4, 1)
    
    for i in range(n):
        user_id = str(uuid.uuid4())
        partner_type = random.choice(SUBSCRIPTION_PLANS["jio_bp"])
        has_equity = random.random() > 0.8
        
        users.append({
            "user_id": user_id,
            "partner_name": f"BP Partner_{i+1:04d}",
            "business_name": f"{random.choice(['AutoCare', 'FuelStation', 'Convenience', 'Fleet'])}_{i+1:04d}",
            "partner_type": partner_type,
            "gstin": f"{random.randint(10,35)}AAAAA{i+1:04d}A1Z5",
            "registration_date": (base_date + timedelta(days=random.randint(0, 400))).date(),
            "is_active": random.random() > 0.1,
            "region": random.choice(REGIONS),
            "state": random.choice(STATES),
            "monthly_transaction_volume_inr": round(random.uniform(50000, 5000000), 2),
            "fuel_dispensed_liters_30d": random.randint(0, 500000),
            "non_fuel_revenue_inr": round(random.uniform(10000, 500000), 2),
            "customer_footfall_30d": random.randint(100, 50000),
            "jio_connectivity_subscriptions": random.randint(1, 100),
            "equity_stake_percentage": round(random.uniform(0.01, 2.0), 4) if has_equity else None,
            "investment_amount_in_cr": round(random.uniform(0.1, 50), 2) if has_equity else None,
            "revenue_share_percentage": round(random.uniform(1, 15), 2),
            "outlet_count": random.randint(1, 50),
            "ev_charging_points": random.randint(0, 20),
            "digital_payment_adoption_pct": round(random.uniform(30, 100), 1),
            "created_at": datetime.now(),
            "updated_at": datetime.now(),
        })
    return users


# ---------------------------------------------------------------------------
# Database Operations (FIXED)
# ---------------------------------------------------------------------------

def create_tables(conn):
    """Create the user/shareholder tables if they don't exist."""
    with conn.cursor() as cur:
        # Reliance Shareholders Table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS reliance_shareholders (
                user_id UUID PRIMARY KEY,
                holder_name VARCHAR(255) NOT NULL,
                holder_type VARCHAR(100),
                pan_number VARCHAR(20),
                demat_account VARCHAR(100),
                shares_held BIGINT,
                share_category VARCHAR(100),
                avg_buy_price DECIMAL(10,2),
                current_value_in_cr DECIMAL(15,4),
                percentage_holding DECIMAL(8,4),
                first_purchase_date DATE,
                last_transaction_date DATE,
                voting_rights BIGINT,
                region VARCHAR(50),
                is_active BOOLEAN DEFAULT TRUE,
                kyc_status VARCHAR(20),
                created_at TIMESTAMPTZ DEFAULT NOW(),
                updated_at TIMESTAMPTZ DEFAULT NOW()
            );
            CREATE INDEX IF NOT EXISTS idx_reliance_holder_type ON reliance_shareholders(holder_type);
            CREATE INDEX IF NOT EXISTS idx_reliance_region ON reliance_shareholders(region);
            CREATE INDEX IF NOT EXISTS idx_reliance_active ON reliance_shareholders(is_active) WHERE is_active = TRUE;
        """)
        
        # Jio Users Table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS jio_users (
                user_id UUID PRIMARY KEY,
                mobile_number VARCHAR(20) UNIQUE,
                email VARCHAR(255),
                subscription_plan VARCHAR(100),
                plan_category VARCHAR(50),
                registration_date DATE,
                last_activity_timestamp TIMESTAMPTZ,
                is_active BOOLEAN DEFAULT TRUE,
                region VARCHAR(50),
                circle VARCHAR(100),
                data_usage_gb_30d DECIMAL(10,2),
                voice_minutes_30d INTEGER,
                sms_count_30d INTEGER,
                arpu_inr DECIMAL(10,2),
                recharge_frequency_days INTEGER,
                linked_jfs_account BOOLEAN DEFAULT FALSE,
                jfs_shares_held INTEGER DEFAULT 0,
                loyalty_points INTEGER DEFAULT 0,
                churn_risk_score DECIMAL(3,3),
                nps_score INTEGER,
                created_at TIMESTAMPTZ DEFAULT NOW(),
                updated_at TIMESTAMPTZ DEFAULT NOW()
            );
            CREATE INDEX IF NOT EXISTS idx_jio_plan ON jio_users(subscription_plan);
            CREATE INDEX IF NOT EXISTS idx_jio_region ON jio_users(region);
            CREATE INDEX IF NOT EXISTS idx_jio_active ON jio_users(is_active) WHERE is_active = TRUE;
        """)
        
        # Jio-BP Users Table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS jio_bp_users (
                user_id UUID PRIMARY KEY,
                partner_name VARCHAR(255) NOT NULL,
                business_name VARCHAR(255),
                partner_type VARCHAR(100),
                gstin VARCHAR(20),
                registration_date DATE,
                is_active BOOLEAN DEFAULT TRUE,
                region VARCHAR(50),
                state VARCHAR(100),
                monthly_transaction_volume_inr DECIMAL(15,2),
                fuel_dispensed_liters_30d INTEGER,
                non_fuel_revenue_inr DECIMAL(12,2),
                customer_footfall_30d INTEGER,
                jio_connectivity_subscriptions INTEGER,
                equity_stake_percentage DECIMAL(6,4),
                investment_amount_in_cr DECIMAL(10,2),
                revenue_share_percentage DECIMAL(5,2),
                outlet_count INTEGER,
                ev_charging_points INTEGER,
                digital_payment_adoption_pct DECIMAL(5,1),
                created_at TIMESTAMPTZ DEFAULT NOW(),
                updated_at TIMESTAMPTZ DEFAULT NOW()
            );
            CREATE INDEX IF NOT EXISTS idx_jiobp_type ON jio_bp_users(partner_type);
            CREATE INDEX IF NOT EXISTS idx_jiobp_region ON jio_bp_users(region);
            CREATE INDEX IF NOT EXISTS idx_jiobp_active ON jio_bp_users(is_active) WHERE is_active = TRUE;
        """)
        
        conn.commit()
        print("✓ Tables created/verified")


def insert_batch_simple(conn, table_name: str, records: List[Dict], batch_size: int = BATCH_SIZE):
    """
    Insert records using execute_batch with simple INSERT (no UPSERT).
    Uses ON CONFLICT DO NOTHING to avoid duplicates on re-runs.
    """
    if not records:
        return
    
    columns = list(records[0].keys())
    placeholders = ", ".join(["%s"] * len(columns))
    col_names = ", ".join(columns)
    
    # Simple INSERT with ON CONFLICT DO NOTHING (safer than complex UPSERT with execute_batch)
    insert_query = f"""
        INSERT INTO {table_name} ({col_names})
        VALUES ({placeholders})
        ON CONFLICT (user_id) DO NOTHING
    """
    
    inserted = 0
    skipped = 0
    
    with conn.cursor() as cur:
        for i in range(0, len(records), batch_size):
            batch = records[i:i+batch_size]
            values = []
            for record in batch:
                row = []
                for col in columns:
                    val = record[col]
                    # Handle special types for psycopg2
                    if isinstance(val, Decimal):
                        row.append(float(val))  # Convert Decimal to float for DB
                    elif isinstance(val, datetime):
                        row.append(val)  # psycopg2 handles datetime natively
                    elif isinstance(val, date):
                        row.append(val)
                    elif val is None:
                        row.append(None)
                    else:
                        row.append(val)
                values.append(tuple(row))
            
            try:
                execute_batch(cur, insert_query, values, page_size=batch_size)
                conn.commit()
                inserted += len(batch)
                print(f"  → Inserted {inserted}/{len(records)} records into {table_name}")
            except Exception as e:
                conn.rollback()
                print(f"  ✗ Error inserting batch into {table_name}: {e}")
                print(f"  → Sample record keys: {list(batch[0].keys()) if batch else 'N/A'}")
                raise
    
    # Count actual inserts vs skips
    with conn.cursor() as cur:
        cur.execute(f"SELECT COUNT(*) FROM {table_name}")
        total = cur.fetchone()[0]
        print(f"  ✓ {table_name}: {total} total records in DB")


def seed_all_data():
    """Main function to generate and insert all seed data."""
    print("🌱 Starting seed data generation...")
    
    conn = None
    try:
        conn = psycopg2.connect(DATABASE_URL)
        print(f"✓ Connected to database")
        
        # Create tables
        create_tables(conn)
        
        # Generate data
        print("\n📊 Generating seed data...")
        reliance_data = generate_reliance_shareholders(n=1000)
        jio_data = generate_jio_users(n=5000)
        jiobp_data = generate_jio_bp_users(n=2000)
        
        print(f"  • Reliance shareholders: {len(reliance_data)} records")
        print(f"  • Jio users: {len(jio_data)} records")  
        print(f"  • Jio-BP partners: {len(jiobp_data)} records")
        
        # Insert data
        print("\n💾 Inserting into database...")
        insert_batch_simple(conn, "reliance_shareholders", reliance_data)
        insert_batch_simple(conn, "jio_users", jio_data)
        insert_batch_simple(conn, "jio_bp_users", jiobp_data)
        
        print("\n✅ Seed data insertion complete!")
        
    except psycopg2.OperationalError as e:
        print(f"\n❌ Database connection error: {e}")
        print(f"💡 Check your ADMIN_DB_URL environment variable")
        raise
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        traceback.print_exc()
        raise
    finally:
        if conn:
            conn.close()
            print("🔌 Database connection closed")


if __name__ == "__main__":
    seed_all_data()