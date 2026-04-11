-- Computer schema draft — throw darts at this
-- 2026-04-11
--
-- Philosophy: bots have instructions and tools (including query access
-- to this database). Everything else is event-driven chaos, just like
-- the real world. No hardcoded routing, no state machines. A bot gets
-- picked, looks around, figures it out.

CREATE TABLE bots (
    sid      TEXT PRIMARY KEY,
    name     TEXT,
    chat     TEXT NOT NULL      -- tool path to invoke for chat routing
);

CREATE TABLE roles (
    name     TEXT PRIMARY KEY,
    bot_sid  TEXT NOT NULL REFERENCES bots(sid),
    location TEXT NOT NULL
);

CREATE TABLE guests (
    username    TEXT PRIMARY KEY,
    first_name  TEXT,
    visit_count INTEGER DEFAULT 0,
    last_visit  TEXT,
    cleared     INTEGER DEFAULT 0,
    location    TEXT DEFAULT 'AtlasLobby'
);

CREATE TABLE games (
    game_id  TEXT PRIMARY KEY,
    username TEXT NOT NULL,
    bot_sid  TEXT NOT NULL REFERENCES bots(sid),
    location TEXT NOT NULL,
    started  TEXT NOT NULL
);

-- Seed data

INSERT INTO bots (sid, name, chat) VALUES
    ('atlas', 'Atlas', 'Bot*Atlas*chat'),
    ('kitty', 'Kitty', 'Bot*Kitty*chat');

INSERT INTO roles (name, bot_sid, location) VALUES
    ('Front Desk Assistant', 'atlas', 'AtlasLobby'),
    ('Receptionist', 'kitty', 'KittyLobby');

