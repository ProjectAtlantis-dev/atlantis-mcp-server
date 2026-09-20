-- TEST DATABASE ONLY. Apply only to a database whose name ends in _test.
BEGIN;

CREATE EXTENSION IF NOT EXISTS pgcrypto;
CREATE SCHEMA IF NOT EXISTS greenland_world;
SET search_path TO greenland_world, public;

CREATE TABLE gl_node (
    code text PRIMARY KEY,
    name text NOT NULL,
    kind text NOT NULL CHECK (kind IN ('port', 'settlement', 'mine', 'airport', 'research_site')),
    lat double precision NOT NULL,
    lon double precision NOT NULL,
    anchor_tile_id text CHECK (anchor_tile_id IS NULL OR anchor_tile_id ~ '^12-[0-9]+-[0-9]+$'),
    services jsonb NOT NULL DEFAULT '[]'::jsonb
);

CREATE TABLE gl_route (
    id text PRIMARY KEY,
    mode text NOT NULL CHECK (mode IN ('sea', 'land', 'air')),
    from_node_code text NOT NULL REFERENCES gl_node(code),
    to_node_code text NOT NULL REFERENCES gl_node(code),
    distance_km numeric(16,6) NOT NULL CHECK (distance_km > 0),
    duration_seconds integer NOT NULL CHECK (duration_seconds > 0),
    risk numeric(8,7) NOT NULL DEFAULT 0 CHECK (risk BETWEEN 0 AND 1),
    path jsonb NOT NULL
);

CREATE TABLE gl_player (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    external_user_id text NOT NULL UNIQUE,
    bank_account_id uuid NOT NULL UNIQUE,
    display_name text NOT NULL,
    home_node_code text NOT NULL REFERENCES gl_node(code),
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE gl_vehicle_state (
    asset_id uuid PRIMARY KEY,
    controller_account_id uuid NOT NULL,
    vehicle_type text NOT NULL,
    status text NOT NULL,
    current_node_code text REFERENCES gl_node(code),
    fuel_liters numeric(20,8) NOT NULL CHECK (fuel_liters >= 0),
    fuel_capacity_liters numeric(20,8) NOT NULL CHECK (fuel_capacity_liters > 0),
    cargo_capacity_tonnes numeric(20,8) NOT NULL CHECK (cargo_capacity_tonnes > 0),
    damage numeric(8,7) NOT NULL DEFAULT 0 CHECK (damage BETWEEN 0 AND 1),
    version bigint NOT NULL DEFAULT 1,
    updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE gl_transit (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    vehicle_asset_id uuid NOT NULL REFERENCES gl_vehicle_state(asset_id),
    controller_account_id uuid NOT NULL,
    route_id text NOT NULL REFERENCES gl_route(id),
    status text NOT NULL CHECK (status IN ('in_transit', 'arrived', 'cancelled', 'distressed')),
    departed_at timestamptz NOT NULL,
    arrives_at timestamptz NOT NULL,
    arrived_at timestamptz,
    fuel_used_liters numeric(20,8) NOT NULL CHECK (fuel_used_liters >= 0),
    path jsonb NOT NULL,
    idempotency_key text NOT NULL UNIQUE,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE gl_cargo_assignment (
    vehicle_asset_id uuid NOT NULL REFERENCES gl_vehicle_state(asset_id),
    resource_asset_id uuid NOT NULL UNIQUE,
    loaded_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (vehicle_asset_id, resource_asset_id)
);

CREATE TABLE gl_market_price (
    node_code text NOT NULL REFERENCES gl_node(code),
    commodity_type text NOT NULL,
    buy_price numeric(28,8) NOT NULL CHECK (buy_price >= 0),
    sell_price numeric(28,8) NOT NULL CHECK (sell_price >= 0),
    stock numeric(28,8) NOT NULL CHECK (stock >= 0),
    updated_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (node_code, commodity_type)
);

CREATE TABLE gl_world_event (
    sequence bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    event_type text NOT NULL,
    actor_account_id uuid,
    entity_id text,
    details jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX gl_transit_active_idx ON gl_transit(arrives_at) WHERE status = 'in_transit';
CREATE INDEX gl_world_event_entity_idx ON gl_world_event(entity_id, sequence);

COMMIT;
