#!/usr/bin/env python3
"""
seed_simple.py — Minimal seed data populator for NorthStar Bank
Usage: 
    export ADMIN_DB_URL="postgresql://postgres:Pass%40123@localhost/agentic_rag_db"
    python seed_simple.py
"""

import os
import sys
from urllib.parse import urlparse, unquote
from dotenv import load_dotenv
load_dotenv(override=True)
try:
    import psycopg
except ImportError:
    print("❌ Install psycopg: pip install 'psycopg[binary]'")
    sys.exit(1)


# ─────────────────────────────────────────────────────────────
# SEED SQL — Schema + Data (Idempotent with ON CONFLICT)
# ─────────────────────────────────────────────────────────────
SEED_SQL = """
-- Extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS vector;

-- 1.1 Accounts
CREATE TABLE IF NOT EXISTS accounts (
    account_id      VARCHAR(20)  PRIMARY KEY,
    customer_name   VARCHAR(100) NOT NULL,
    account_type    VARCHAR(20)  NOT NULL CHECK (account_type IN ('savings','current','salary')),
    branch_code     VARCHAR(10)  NOT NULL,
    ifsc_code       VARCHAR(15),
    mobile          VARCHAR(15),
    email           VARCHAR(100),
    kyc_status      VARCHAR(20)  DEFAULT 'verified',
    created_at      TIMESTAMP    DEFAULT NOW()
);

-- 1.2 Transactions
CREATE TABLE IF NOT EXISTS transactions (
    txn_id          UUID         PRIMARY KEY DEFAULT uuid_generate_v4(),
    account_id      VARCHAR(20)  REFERENCES accounts(account_id),
    txn_date        DATE         NOT NULL,
    txn_type        VARCHAR(10)  NOT NULL CHECK (txn_type IN ('debit','credit')),
    amount          NUMERIC(15,2) NOT NULL,
    balance_after   NUMERIC(15,2),
    description     VARCHAR(200),
    channel         VARCHAR(20)  CHECK (channel IN ('ATM','UPI','NEFT','RTGS','IMPS','branch','online','POS')),
    merchant_name   VARCHAR(100),
    category        VARCHAR(50),
    created_at      TIMESTAMP    DEFAULT NOW()
);

-- 1.3 Loan Accounts
CREATE TABLE IF NOT EXISTS loan_accounts (
    loan_id         VARCHAR(20)  PRIMARY KEY,
    account_id      VARCHAR(20)  REFERENCES accounts(account_id),
    loan_type       VARCHAR(30)  NOT NULL CHECK (loan_type IN ('home_loan','personal_loan','auto_loan','gold_loan')),
    principal       NUMERIC(15,2) NOT NULL,
    outstanding     NUMERIC(15,2) NOT NULL,
    disbursed_date  DATE,
    emi_amount      NUMERIC(15,2),
    next_emi_date   DATE,
    interest_rate   NUMERIC(5,2),
    tenure_months   INT,
    emi_paid        INT DEFAULT 0,
    status          VARCHAR(20)  DEFAULT 'active',
    created_at      TIMESTAMP    DEFAULT NOW()
);

-- 1.4 Fixed Deposits
CREATE TABLE IF NOT EXISTS fixed_deposits (
    fd_id           VARCHAR(20)  PRIMARY KEY,
    account_id      VARCHAR(20)  REFERENCES accounts(account_id),
    principal       NUMERIC(15,2) NOT NULL,
    interest_rate   NUMERIC(5,2) NOT NULL,
    tenure_days     INT          NOT NULL,
    start_date      DATE         NOT NULL,
    maturity_date   DATE         NOT NULL,
    maturity_amount NUMERIC(15,2),
    interest_payout VARCHAR(20)  DEFAULT 'at_maturity',
    status          VARCHAR(20)  DEFAULT 'active',
    created_at      TIMESTAMP    DEFAULT NOW()
);

-- 1.5 Credit Cards
CREATE TABLE IF NOT EXISTS credit_cards (
    card_id         VARCHAR(20)  PRIMARY KEY,
    account_id      VARCHAR(20)  REFERENCES accounts(account_id),
    card_variant    VARCHAR(30),
    credit_limit    NUMERIC(15,2),
    available_limit NUMERIC(15,2),
    outstanding_amt NUMERIC(15,2) DEFAULT 0,
    due_date        DATE,
    min_due         NUMERIC(15,2) DEFAULT 0,
    status          VARCHAR(20)  DEFAULT 'active',
    issued_date     DATE,
    created_at      TIMESTAMP    DEFAULT NOW()
);

-- 1.6 Credit Card Transactions
CREATE TABLE IF NOT EXISTS card_transactions (
    txn_id          UUID         PRIMARY KEY DEFAULT uuid_generate_v4(),
    card_id         VARCHAR(20)  REFERENCES credit_cards(card_id),
    txn_date        DATE         NOT NULL,
    txn_type        VARCHAR(20)  CHECK (txn_type IN ('purchase','cashadvance','payment','refund','fee')),
    amount          NUMERIC(15,2) NOT NULL,
    merchant_name   VARCHAR(100),
    category        VARCHAR(50),
    is_international BOOLEAN     DEFAULT FALSE,
    currency        VARCHAR(5)   DEFAULT 'INR',
    created_at      TIMESTAMP    DEFAULT NOW()
);

-- SEED: Accounts
INSERT INTO accounts VALUES
('1345367','James Mitchell','savings','CHN001','NSBK0CHN001','XXXXXXX890','james.mitchell@email.com','verified','2019-03-15 10:00:00'),
('2456789','Sarah Thompson','salary','MUM002','NSBK0MUM002','XXXXXXX123','sarah.thompson@email.com','verified','2020-07-22 11:30:00'),
('3567890','Robert Clarke','current','DEL003','NSBK0DEL003','XXXXXXX456','robert.clarke@bizmail.com','verified','2018-11-05 09:15:00'),
('4678901','Emily Watson','savings','HYD004','NSBK0HYD004','XXXXXXX789','emily.watson@email.com','verified','2021-02-18 14:45:00'),
('5789012','Daniel Foster','salary','BLR005','NSBK0BLR005','XXXXXXX321','daniel.foster@email.com','verified','2022-06-01 08:00:00'),
('6890123','Laura Bennett','savings','CHN001','NSBK0CHN001','XXXXXXX654','laura.bennett@email.com','verified','2020-09-30 16:20:00'),
('7901234','Michael Harrington','current','DEL003','NSBK0DEL003','XXXXXXX987','michael.harrington@corp.com','verified','2017-04-10 11:00:00'),
('8012345','Catherine Ellis','savings','BLR005','NSBK0BLR005','XXXXXXX147','catherine.ellis@email.com','verified','2023-01-14 09:30:00')
ON CONFLICT (account_id) DO UPDATE SET customer_name=EXCLUDED.customer_name, email=EXCLUDED.email;

-- SEED: Transactions (James Mitchell - 1345367)
INSERT INTO transactions (account_id, txn_date, txn_type, amount, balance_after, description, channel, merchant_name, category) VALUES
('1345367','2026-01-03','credit',85000.00,185000.00,'Salary Credit - January 2026','NEFT','Apex Solutions Inc','salary'),
('1345367','2026-01-05','debit',15000.00,170000.00,'Home Loan EMI - January','NEFT','NorthStar Bank','loan_emi'),
('1345367','2026-01-07','debit',4500.00,165500.00,'BigBasket Online Grocery','UPI','BigBasket','groceries'),
('1345367','2026-01-10','debit',2200.00,163300.00,'Swiggy Food Order','UPI','Swiggy','food_dining'),
('1345367','2026-01-12','debit',12000.00,151300.00,'Amazon Shopping','online','Amazon India','shopping'),
('1345367','2026-01-15','debit',3500.00,147800.00,'BESCOM Electricity Bill','online','BESCOM','utilities'),
('1345367','2026-01-18','debit',8000.00,139800.00,'Tata Motors Service Center','POS','Tata Service','automobile'),
('1345367','2026-01-20','credit',5000.00,144800.00,'UPI Transfer Received - Kevin Walsh','UPI',NULL,'transfer'),
('1345367','2026-01-22','debit',1800.00,143000.00,'Netflix Subscription','online','Netflix','entertainment'),
('1345367','2026-01-25','debit',25000.00,118000.00,'HDFC Life Insurance Premium','NEFT','HDFC Life','insurance'),
('1345367','2026-01-28','debit',6700.00,111300.00,'Apollo Pharmacy','UPI','Apollo Pharmacy','medical'),
('1345367','2026-01-31','debit',3200.00,108100.00,'Airtel Postpaid Bill','online','Airtel','utilities'),
('1345367','2026-02-03','credit',85000.00,193100.00,'Salary Credit - February 2026','NEFT','Apex Solutions Inc','salary'),
('1345367','2026-02-05','debit',15000.00,178100.00,'Home Loan EMI - February','NEFT','NorthStar Bank','loan_emi'),
('1345367','2026-02-08','debit',5200.00,172900.00,'Reliance Smart Grocery','POS','Reliance Smart','groceries'),
('1345367','2026-02-10','debit',18500.00,154400.00,'Croma Electronics - Headphones','POS','Croma','electronics'),
('1345367','2026-02-14','debit',3800.00,150600.00,'Zomato Valentine Dinner','UPI','Zomato','food_dining'),
('1345367','2026-02-15','debit',2800.00,147800.00,'BWSSB Water Bill','online','BWSSB','utilities'),
('1345367','2026-02-18','debit',55000.00,92800.00,'NEFT to James FD Account','NEFT',NULL,'transfer'),
('1345367','2026-02-20','debit',4100.00,88700.00,'Ola Cabs Monthly Pass','UPI','Ola','transport'),
('1345367','2026-02-22','debit',9800.00,78900.00,'Decathlon Sports Equipment','POS','Decathlon','shopping'),
('1345367','2026-02-25','debit',1200.00,77700.00,'Hotstar Annual Subscription','online','Disney+ Hotstar','entertainment'),
('1345367','2026-02-28','debit',7500.00,70200.00,'Dr. Rajan Clinic Consultation','UPI','Apollo Clinic','medical'),
('1345367','2026-03-03','credit',85000.00,155200.00,'Salary Credit - March 2026','NEFT','Apex Solutions Inc','salary'),
('1345367','2026-03-05','debit',15000.00,140200.00,'Home Loan EMI - March','NEFT','NorthStar Bank','loan_emi'),
('1345367','2026-03-07','debit',6300.00,133900.00,'More Supermarket','POS','More Supermarket','groceries'),
('1345367','2026-03-10','debit',32000.00,101900.00,'Flight Booking - IndiGo','online','IndiGo Airlines','travel'),
('1345367','2026-03-12','debit',15000.00,86900.00,'MakeMyTrip Hotel Booking','online','MakeMyTrip','travel'),
('1345367','2026-03-15','debit',3500.00,83400.00,'BESCOM Electricity Bill','online','BESCOM','utilities'),
('1345367','2026-03-17','debit',2500.00,80900.00,'Swiggy Food Delivery','UPI','Swiggy','food_dining'),
('1345367','2026-03-19','debit',75000.00,5900.00,'NEFT - Advance Tax Q4','NEFT','Income Tax Dept','tax'),
('1345367','2026-03-20','credit',75000.00,80900.00,'UPI Transfer Received - Bonus','UPI',NULL,'transfer'),
('1345367','2026-03-22','debit',4800.00,76100.00,'Nykaa Shopping','online','Nykaa','shopping'),
('1345367','2026-03-25','debit',12000.00,64100.00,'LIC Premium Payment','online','LIC India','insurance'),
('1345367','2026-03-28','debit',3300.00,60800.00,'Airtel Postpaid Bill','online','Airtel','utilities'),
('1345367','2026-03-31','debit',8500.00,52300.00,'D-Mart Shopping','POS','D-Mart','groceries'),
('1345367','2026-04-01','credit',85000.00,137300.00,'Salary Credit - April 2026','NEFT','Apex Solutions Inc','salary'),
('1345367','2026-04-05','debit',15000.00,122300.00,'Home Loan EMI - April','NEFT','NorthStar Bank','loan_emi'),
('1345367','2026-04-07','debit',5500.00,116800.00,'BigBasket Online Grocery','UPI','BigBasket','groceries'),
('1345367','2026-04-10','debit',3100.00,113700.00,'Zomato Order','UPI','Zomato','food_dining'),
('2456789','2026-03-01','credit',120000.00,250000.00,'Salary Credit March 2026','NEFT','GlobalTech Corp','salary'),
('2456789','2026-03-05','debit',22000.00,228000.00,'Home Loan EMI','NEFT','NorthStar Bank','loan_emi'),
('2456789','2026-03-10','debit',8500.00,219500.00,'Amazon Shopping','online','Amazon India','shopping'),
('2456789','2026-03-15','debit',4200.00,215300.00,'Grocery - Spencers','POS','Spencers Retail','groceries'),
('2456789','2026-04-01','credit',120000.00,335300.00,'Salary Credit April 2026','NEFT','GlobalTech Corp','salary'),
('2456789','2026-04-05','debit',22000.00,313300.00,'Home Loan EMI','NEFT','NorthStar Bank','loan_emi')
ON CONFLICT ON CONSTRAINT transactions_pkey DO NOTHING;

-- SEED: Loan Accounts
INSERT INTO loan_accounts VALUES
('L-789012','1345367','home_loan',4500000.00,3820000.00,'2021-06-15',15000.00,'2026-05-05',8.75,300,58,'active'),
('L-892345','2456789','home_loan',6000000.00,5400000.00,'2023-01-20',22000.00,'2026-05-05',8.50,360,39,'active'),
('L-345678','4678901','personal_loan',500000.00,280000.00,'2024-03-10',9800.00,'2026-05-10',13.50,60,25,'active'),
('L-456789','5789012','auto_loan',1200000.00,950000.00,'2023-09-01',24500.00,'2026-05-01',9.25,60,19,'active'),
('L-567890','7901234','home_loan',8000000.00,7200000.00,'2022-11-05',35000.00,'2026-05-05',9.00,360,41,'active'),
('L-678901','1345367','personal_loan',300000.00,0.00,'2020-01-15',6500.00,NULL,12.50,48,48,'closed')
ON CONFLICT (loan_id) DO UPDATE SET outstanding=EXCLUDED.outstanding, status=EXCLUDED.status;

-- SEED: Fixed Deposits
INSERT INTO fixed_deposits VALUES
('FD-111001','1345367',200000.00,7.25,730,'2025-02-18','2027-02-18',232900.00,'at_maturity','active'),
('FD-111002','1345367',50000.00,7.10,365,'2026-01-01','2027-01-01',53550.00,'quarterly','active'),
('FD-222001','2456789',500000.00,7.50,444,'2025-11-01','2027-01-19',546250.00,'at_maturity','active'),
('FD-333001','3567890',100000.00,6.75,548,'2024-06-01','2025-12-01',110125.00,'quarterly','matured'),
('FD-444001','4678901',150000.00,7.25,730,'2025-09-10','2027-09-10',172350.00,'at_maturity','active'),
('FD-555001','6890123',75000.00,7.00,365,'2026-03-01','2027-03-01',80250.00,'at_maturity','active')
ON CONFLICT (fd_id) DO UPDATE SET status=EXCLUDED.status;

-- SEED: Credit Cards
INSERT INTO credit_cards VALUES
('CC-881001','1345367','NorthStar Gold',200000.00,145000.00,55000.00,'2026-04-25',2750.00,'active','2021-07-01'),
('CC-882001','2456789','NorthStar Platinum',500000.00,420000.00,80000.00,'2026-04-28',4000.00,'active','2022-03-15'),
('CC-883001','5789012','NorthStar Classic',75000.00,60000.00,15000.00,'2026-04-20',750.00,'active','2023-01-10'),
('CC-884001','8012345','NorthStar Signature',1000000.00,850000.00,150000.00,'2026-04-30',7500.00,'active','2024-02-01')
ON CONFLICT (card_id) DO UPDATE SET available_limit=EXCLUDED.available_limit, outstanding_amt=EXCLUDED.outstanding_amt;

-- SEED: Credit Card Transactions
INSERT INTO card_transactions (card_id, txn_date, txn_type, amount, merchant_name, category, is_international, currency) VALUES
('CC-881001','2026-03-02','purchase',3200.00,'Barbeque Nation','food_dining',FALSE,'INR'),
('CC-881001','2026-03-05','purchase',12000.00,'Myntra','shopping',FALSE,'INR'),
('CC-881001','2026-03-10','purchase',8500.00,'Marriott Hotels','travel',FALSE,'INR'),
('CC-881001','2026-03-14','purchase',28500.00,'Singapore Airlines','travel',TRUE,'SGD'),
('CC-881001','2026-03-15','purchase',4200.00,'Amazon UK','shopping',TRUE,'GBP'),
('CC-881001','2026-03-18','purchase',1500.00,'Spotify','entertainment',FALSE,'INR'),
('CC-881001','2026-03-22','purchase',9800.00,'Tanishq Jewellery','jewellery',FALSE,'INR'),
('CC-881001','2026-03-25','fee',340.00,'NorthStar Bank','bank_fee',FALSE,'INR'),
('CC-881001','2026-04-01','payment',30000.00,'NorthStar Payment','payment',FALSE,'INR'),
('CC-881001','2026-04-03','purchase',6700.00,'Reliance Digital','electronics',FALSE,'INR'),
('CC-881001','2026-04-07','purchase',2100.00,'Dominos Pizza','food_dining',FALSE,'INR'),
('CC-881001','2026-04-10','purchase',18000.00,'IRCTC Tatkal Ticket','travel',FALSE,'INR')
ON CONFLICT ON CONSTRAINT card_transactions_pkey DO NOTHING;
"""


def parse_db_url(url: str) -> dict:
    """Parse PostgreSQL URL into connection parameters."""
    parsed = urlparse(url)
    password = unquote(parsed.password) if parsed.password else ""
    return {
        "host": parsed.hostname or "localhost",
        "port": parsed.port or 5432,
        "dbname": parsed.path.lstrip("/") or "postgres",
        "user": parsed.username or "postgres",
        "password": password,
    }


def main():
    # Get DB URL from environment
    db_url = os.getenv("ADMIN_DB_URL")
    if not db_url:
        print("❌ Please set ADMIN_DB_URL environment variable")
        print('   Example: export ADMIN_DB_URL="postgresql://user:pass@localhost/dbname"')
        sys.exit(1)

    # Parse and connect
    config = parse_db_url(db_url)
    print(f"🔌 Connecting to {config['host']}/{config['dbname']} as {config['user']}...")

    try:
        conn = psycopg.connect(
            host=config["host"],
            port=config["port"],
            dbname=config["dbname"],
            user=config["user"],
            password=config["password"],
        )
        print("✅ Connected successfully")

        # Execute seed SQL
        print("🚀 Executing seed data...")
        with conn.cursor() as cur:
            cur.execute(SEED_SQL)
        conn.commit()
        print("✅ Seed data populated successfully!")

        # Quick verification
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM accounts")
            accounts = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM transactions")
            txns = cur.fetchone()[0]
            print(f"📊 Verification: {accounts} accounts, {txns} transactions")

        conn.close()

    except psycopg.OperationalError as e:
        print(f"❌ Connection failed: {e}")
        print("💡 Check: Is PostgreSQL running? Are credentials correct?")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()