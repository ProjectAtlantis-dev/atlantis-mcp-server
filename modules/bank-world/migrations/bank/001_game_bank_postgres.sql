BEGIN;

CREATE EXTENSION IF NOT EXISTS pgcrypto;
CREATE SCHEMA IF NOT EXISTS greenland_game;
SET search_path TO greenland_game, public;

CREATE TABLE gl_bank_account (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    display_name text NOT NULL,
    account_type text NOT NULL CHECK (account_type IN ('player', 'warehouse', 'treasury', 'system')),
    created_at timestamptz NOT NULL DEFAULT now()
);

-- Game identity maps to x_user.id but game balances never use x_user billing rows.
-- Add a foreign key to public.x_user(id) only when both schemas share one database.
CREATE TABLE gl_player (
    x_user_id bigint PRIMARY KEY,
    bank_account_id uuid NOT NULL UNIQUE REFERENCES gl_bank_account(id),
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE gl_service_identity (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    service_name text NOT NULL UNIQUE,
    service_type text NOT NULL CHECK (service_type IN ('warehouse', 'game_authority')),
    owner_account_id uuid NOT NULL REFERENCES gl_bank_account(id),
    token_hash text NOT NULL UNIQUE,
    is_active boolean NOT NULL DEFAULT true,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE gl_bank_transaction (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    transaction_type text NOT NULL,
    idempotency_key text UNIQUE,
    actor_account_id uuid REFERENCES gl_bank_account(id),
    status text NOT NULL DEFAULT 'posted' CHECK (status IN ('posted', 'reversed')),
    details jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE gl_bank_entry (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    transaction_id uuid NOT NULL REFERENCES gl_bank_transaction(id),
    account_id uuid NOT NULL REFERENCES gl_bank_account(id),
    currency text NOT NULL DEFAULT 'GLC',
    amount numeric(28,8) NOT NULL CHECK (amount <> 0),
    entry_role text NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE gl_asset (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    kind text NOT NULL CHECK (kind IN (
        'resource_lot', 'vehicle', 'equipment', 'machinery', 'powerplant',
        'warehouse', 'structure', 'land_title'
    )),
    asset_type text NOT NULL,
    owner_account_id uuid NOT NULL REFERENCES gl_bank_account(id),
    custodian_account_id uuid REFERENCES gl_bank_account(id),
    quantity numeric(28,8),
    unit text,
    status text NOT NULL DEFAULT 'active' CHECK (status IN (
        'active', 'stored', 'in_transit', 'installed', 'deployed', 'consumed', 'retired', 'destroyed'
    )),
    origin_tile_id text,
    location_tile_id text,
    metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
    version bigint NOT NULL DEFAULT 1,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now(),
    CHECK ((kind = 'resource_lot' AND quantity > 0 AND unit IS NOT NULL)
        OR (kind <> 'resource_lot' AND quantity IS NULL))
);

CREATE TABLE gl_parcel (
    tile_id text PRIMARY KEY CHECK (tile_id ~ '^12-[0-9]+-[0-9]+$'),
    depth smallint NOT NULL DEFAULT 12 CHECK (depth = 12),
    col integer NOT NULL,
    row integer NOT NULL,
    title_asset_id uuid NOT NULL UNIQUE REFERENCES gl_asset(id),
    sale_status text NOT NULL DEFAULT 'held' CHECK (sale_status IN ('held', 'for_sale', 'owned')),
    price_amount numeric(28,8),
    price_currency text NOT NULL DEFAULT 'GLC',
    metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    CHECK ((sale_status <> 'for_sale') OR price_amount > 0)
);

ALTER TABLE gl_asset
    ADD CONSTRAINT gl_asset_origin_parcel_fk FOREIGN KEY (origin_tile_id) REFERENCES gl_parcel(tile_id)
        DEFERRABLE INITIALLY DEFERRED,
    ADD CONSTRAINT gl_asset_location_parcel_fk FOREIGN KEY (location_tile_id) REFERENCES gl_parcel(tile_id)
        DEFERRABLE INITIALLY DEFERRED;

CREATE TABLE gl_asset_event (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    asset_id uuid NOT NULL REFERENCES gl_asset(id),
    transaction_id uuid NOT NULL REFERENCES gl_bank_transaction(id),
    event_type text NOT NULL,
    actor_account_id uuid REFERENCES gl_bank_account(id),
    details jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE gl_asset_lineage (
    parent_asset_id uuid NOT NULL REFERENCES gl_asset(id),
    child_asset_id uuid NOT NULL REFERENCES gl_asset(id),
    quantity numeric(28,8),
    created_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (parent_asset_id, child_asset_id),
    CHECK (parent_asset_id <> child_asset_id)
);

CREATE TABLE gl_warehouse_quote (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    client_quote_id text NOT NULL,
    warehouse_account_id uuid NOT NULL REFERENCES gl_bank_account(id),
    seller_account_id uuid NOT NULL REFERENCES gl_bank_account(id),
    asset_ids uuid[] NOT NULL CHECK (cardinality(asset_ids) BETWEEN 1 AND 100),
    listing_ids uuid[] NOT NULL CHECK (cardinality(listing_ids) = cardinality(asset_ids)),
    gross_amount numeric(28,8) NOT NULL CHECK (gross_amount > 0),
    commission_amount numeric(28,8) NOT NULL DEFAULT 0
        CHECK (commission_amount >= 0 AND commission_amount <= gross_amount),
    currency text NOT NULL DEFAULT 'GLC',
    status text NOT NULL DEFAULT 'open' CHECK (status IN ('open', 'settled', 'expired', 'cancelled')),
    expires_at timestamptz NOT NULL,
    transaction_id uuid UNIQUE REFERENCES gl_bank_transaction(id),
    terms jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE (warehouse_account_id, client_quote_id)
);

CREATE TABLE gl_warehouse_listing (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    idempotency_key text NOT NULL UNIQUE,
    asset_id uuid NOT NULL REFERENCES gl_asset(id),
    seller_account_id uuid NOT NULL REFERENCES gl_bank_account(id),
    warehouse_account_id uuid NOT NULL REFERENCES gl_bank_account(id),
    minimum_gross_amount numeric(28,8) NOT NULL DEFAULT 0 CHECK (minimum_gross_amount >= 0),
    currency text NOT NULL DEFAULT 'GLC',
    status text NOT NULL DEFAULT 'open' CHECK (status IN ('open', 'sold', 'expired', 'cancelled')),
    expires_at timestamptz NOT NULL,
    transaction_id uuid REFERENCES gl_bank_transaction(id),
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE gl_production_rule (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    producer_asset_type text NOT NULL,
    output_asset_type text NOT NULL,
    output_unit text NOT NULL,
    quantity_per_hour numeric(28,8) NOT NULL CHECK (quantity_per_hour > 0),
    is_active boolean NOT NULL DEFAULT true,
    metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE gl_production_claim (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    transaction_id uuid NOT NULL UNIQUE REFERENCES gl_bank_transaction(id),
    idempotency_key text NOT NULL UNIQUE,
    source_event_id text NOT NULL UNIQUE,
    rule_id uuid NOT NULL REFERENCES gl_production_rule(id),
    producer_asset_id uuid NOT NULL REFERENCES gl_asset(id),
    owner_account_id uuid NOT NULL REFERENCES gl_bank_account(id),
    origin_tile_id text NOT NULL REFERENCES gl_parcel(tile_id),
    elapsed_seconds integer NOT NULL CHECK (elapsed_seconds BETWEEN 1 AND 86400),
    output_asset_id uuid NOT NULL UNIQUE REFERENCES gl_asset(id),
    evidence jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE gl_recipe (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    recipe_name text NOT NULL UNIQUE,
    processor_asset_type text NOT NULL,
    inputs jsonb NOT NULL,
    outputs jsonb NOT NULL,
    is_active boolean NOT NULL DEFAULT true,
    metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE gl_transformation (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    transaction_id uuid NOT NULL UNIQUE REFERENCES gl_bank_transaction(id),
    idempotency_key text NOT NULL UNIQUE,
    source_event_id text NOT NULL UNIQUE,
    recipe_id uuid NOT NULL REFERENCES gl_recipe(id),
    processor_asset_id uuid NOT NULL REFERENCES gl_asset(id),
    owner_account_id uuid NOT NULL REFERENCES gl_bank_account(id),
    origin_tile_id text NOT NULL REFERENCES gl_parcel(tile_id),
    batches numeric(28,8) NOT NULL CHECK (batches > 0),
    input_asset_ids uuid[] NOT NULL,
    output_asset_ids uuid[] NOT NULL,
    evidence jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX gl_entry_account_idx ON gl_bank_entry(account_id, currency);
CREATE INDEX gl_entry_transaction_idx ON gl_bank_entry(transaction_id);
CREATE INDEX gl_asset_owner_idx ON gl_asset(owner_account_id);
CREATE INDEX gl_asset_custodian_idx ON gl_asset(custodian_account_id);
CREATE INDEX gl_asset_origin_idx ON gl_asset(origin_tile_id);
CREATE INDEX gl_asset_location_idx ON gl_asset(location_tile_id);
CREATE INDEX gl_asset_event_idx ON gl_asset_event(asset_id, created_at);
CREATE INDEX gl_lineage_child_idx ON gl_asset_lineage(child_asset_id);
CREATE INDEX gl_quote_open_idx ON gl_warehouse_quote(expires_at) WHERE status = 'open';
CREATE INDEX gl_listing_asset_idx ON gl_warehouse_listing(asset_id, status, expires_at);

CREATE FUNCTION gl_assert_balanced_transaction() RETURNS trigger
LANGUAGE plpgsql AS $$
DECLARE
    target_transaction uuid := COALESCE(NEW.transaction_id, OLD.transaction_id);
    ledger_sum numeric(28,8);
BEGIN
    SELECT COALESCE(sum(amount), 0) INTO ledger_sum
      FROM gl_bank_entry WHERE transaction_id = target_transaction;
    IF ledger_sum <> 0 THEN
        RAISE EXCEPTION 'Unbalanced game-bank transaction %, sum=%', target_transaction, ledger_sum;
    END IF;
    RETURN NULL;
END;
$$;

CREATE CONSTRAINT TRIGGER gl_bank_entry_balanced
AFTER INSERT OR UPDATE OR DELETE ON gl_bank_entry
DEFERRABLE INITIALLY DEFERRED
FOR EACH ROW EXECUTE FUNCTION gl_assert_balanced_transaction();

CREATE FUNCTION gl_reject_mutation() RETURNS trigger
LANGUAGE plpgsql AS $$
BEGIN
    RAISE EXCEPTION '% is append-only; post a reversing transaction/event instead', TG_TABLE_NAME;
END;
$$;

CREATE TRIGGER gl_transaction_append_only
BEFORE UPDATE OR DELETE ON gl_bank_transaction
FOR EACH ROW EXECUTE FUNCTION gl_reject_mutation();

CREATE TRIGGER gl_entry_append_only
BEFORE UPDATE OR DELETE ON gl_bank_entry
FOR EACH ROW EXECUTE FUNCTION gl_reject_mutation();

CREATE TRIGGER gl_asset_event_append_only
BEFORE UPDATE OR DELETE ON gl_asset_event
FOR EACH ROW EXECUTE FUNCTION gl_reject_mutation();

COMMIT;
