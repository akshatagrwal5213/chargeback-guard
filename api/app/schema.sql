-- chargeback-guard schema. Idempotent: safe to re-run.
-- Every table here exists to be *retrieved from* by the evidence agent.
-- If a table cannot be cited in a dispute response, it does not belong in this file.

create extension if not exists "uuid-ossp";

-- ---------------------------------------------------------------- customers

create table if not exists customers (
    id              text primary key,
    email           text,
    email_domain    text,
    created_at      timestamptz not null default now(),
    account_age_days integer,
    lifetime_value  numeric(12, 2) default 0,
    prior_disputes  integer default 0,
    is_guest        boolean default false
);

-- ------------------------------------------------------------------- orders
-- One row per transaction. `disputed` is the training label for the
-- propensity model; `disputed_at` is what makes a time-based split possible.

create table if not exists orders (
    id                  text primary key,
    customer_id         text references customers (id) on delete set null,
    created_at          timestamptz not null,
    amount              numeric(12, 2) not null,
    currency            text not null default 'INR',
    product_code        text,
    is_digital          boolean default false,

    -- signals captured at order time
    ip                  inet,
    device_fingerprint  text,
    billing_country     text,
    shipping_country    text,
    bin_country         text,
    avs_result          text,           -- Y / N / P / U
    cvv_result          text,           -- M / N / P / U
    three_ds_status     text,           -- authenticated / attempted / failed / none
    statement_descriptor text,

    -- processor linkage
    rail                text not null default 'stripe',   -- stripe | razorpay
    processor_payment_id text,

    -- label
    disputed            boolean default false,
    disputed_at         timestamptz,

    -- set by the propensity model; drives pre-emptive evidence capture
    evidence_tier       text default 'standard'           -- standard | enhanced
);

create index if not exists orders_created_at_idx on orders (created_at);
create index if not exists orders_customer_idx on orders (customer_id);
create index if not exists orders_disputed_idx on orders (disputed);

-- ------------------------------------------------------------- order scores

create table if not exists order_scores (
    order_id        text primary key references orders (id) on delete cascade,
    score           double precision not null,       -- calibrated P(chargeback)
    raw_score       double precision,                -- pre-calibration
    model_version   text not null,
    reasons         jsonb not null default '[]',     -- top SHAP contributions
    scored_at       timestamptz not null default now()
);

-- ================================================================
--  EVIDENCE TABLES
--  These five are what the agent's tools read. Everything the system
--  asserts in a dispute response must come from one of them.
-- ================================================================

create table if not exists fulfillment_events (
    id              bigserial primary key,
    order_id        text not null references orders (id) on delete cascade,
    carrier         text,
    tracking_number text,
    event_type      text not null,          -- label_created | in_transit | out_for_delivery | delivered | exception
    occurred_at     timestamptz not null,
    location        text,
    signature_name  text,
    raw             jsonb
);

create index if not exists fulfillment_order_idx on fulfillment_events (order_id, occurred_at);

create table if not exists communications (
    id              bigserial primary key,
    customer_id     text references customers (id) on delete cascade,
    order_id        text references orders (id) on delete cascade,
    channel         text not null,          -- email | chat | ticket | phone_note
    direction       text not null,          -- inbound | outbound
    occurred_at     timestamptz not null,
    subject         text,
    body            text not null,
    agent_name      text
);

create index if not exists comms_order_idx on communications (order_id, occurred_at);
create index if not exists comms_customer_idx on communications (customer_id, occurred_at);

create table if not exists policy_acceptances (
    id              bigserial primary key,
    order_id        text not null references orders (id) on delete cascade,
    policy_type     text not null,          -- terms | refund | cancellation | shipping
    policy_version  text not null,
    policy_url      text,
    policy_text     text,
    accepted_at     timestamptz not null,
    accepted_ip     inet
);

create index if not exists policy_order_idx on policy_acceptances (order_id);

create table if not exists refunds (
    id              text primary key,
    order_id        text not null references orders (id) on delete cascade,
    amount          numeric(12, 2) not null,
    status          text not null,          -- requested | issued | declined
    reason          text,
    requested_at    timestamptz,
    issued_at       timestamptz,
    processor_refund_id text
);

create index if not exists refunds_order_idx on refunds (order_id);

create table if not exists access_logs (
    id              bigserial primary key,
    order_id        text not null references orders (id) on delete cascade,
    customer_id     text references customers (id) on delete cascade,
    occurred_at     timestamptz not null,
    ip              inet,
    user_agent      text,
    action          text not null           -- login | download | stream | api_call
);

create index if not exists access_order_idx on access_logs (order_id, occurred_at);

-- ----------------------------------------------------------------- disputes

create table if not exists disputes (
    id                  text primary key,          -- our id
    -- Nullable on purpose. A dispute from the processor is authoritative and
    -- must be recorded even when no matching local order exists — the payment
    -- may predate this system, or come from another channel.
    order_id            text references orders (id) on delete set null,
    rail                text not null,             -- stripe | razorpay
    processor_dispute_id text unique,
    -- Where this dispute came from. 'processor' means it arrived over a
    -- webhook and exists at Stripe/Razorpay; 'synthetic' means we generated
    -- it. Only the first can be filed. Recorded rather than inferred from the
    -- id: a generated id can be made to look like anything.
    origin              text not null default 'synthetic',

    amount              numeric(12, 2) not null,
    currency            text not null default 'INR',
    category            text not null,             -- our normalised category
    processor_reason    text,                      -- raw reason from the rail
    network_code        text,                      -- e.g. visa 13.1
    phase               text default 'chargeback', -- razorpay ladder; stripe -> chargeback
    status              text not null default 'needs_response',
    respond_by          timestamptz,
    opened_at           timestamptz not null default now(),

    -- triage output
    win_probability     double precision,
    expected_value      numeric(12, 2),
    recommendation      text,                      -- contest | accept
    triage_rule_version text,
    triaged_at          timestamptz,

    raw_payload         jsonb
);

create index if not exists disputes_status_idx on disputes (status);
create index if not exists disputes_respond_by_idx on disputes (respond_by);

-- ------------------------------------------------------- generated packets

create table if not exists evidence_packets (
    id              uuid primary key default uuid_generate_v4(),
    dispute_id      text not null references disputes (id) on delete cascade,
    created_at      timestamptz not null default now(),
    agent_model     text,

    -- our internal, rail-agnostic evidence object
    evidence        jsonb not null default '{}',
    narrative       text,

    -- checklist outcome
    slots_required  integer default 0,
    slots_filled    integer default 0,
    gaps            jsonb not null default '[]',

    -- the citation guard's verdict on this draft. Stored, not recomputed:
    -- whether a packet may be filed is a fact about the document that was
    -- written, and re-running the guard later would judge a different draft.
    guard           jsonb not null default '{}',

    pdf_url         text,
    submitted_at    timestamptz,
    submission_response jsonb
);

create index if not exists packets_dispute_idx on evidence_packets (dispute_id);

-- The citation guard lives here. One row per factual claim in the narrative.
-- A packet with an uncited claim must not render.
create table if not exists evidence_citations (
    id              bigserial primary key,
    packet_id       uuid not null references evidence_packets (id) on delete cascade,
    claim           text not null,
    source_table    text not null,
    source_id       text not null,
    source_field    text,
    verified        boolean not null default false
);

create index if not exists citations_packet_idx on evidence_citations (packet_id);

-- ------------------------------------------------- outcomes + similarity

-- Real outcomes land here. The triage rule retrains off this table once
-- there are enough rows; until then it stays an explicit calibrated prior.
create table if not exists dispute_outcomes (
    dispute_id      text primary key references disputes (id) on delete cascade,
    outcome         text not null,          -- won | lost | accepted | expired
    closed_at       timestamptz not null default now(),
    amount_recovered numeric(12, 2) default 0,
    notes           text
);

-- ---------------------------------------------------------------- run log

create table if not exists agent_runs (
    id              uuid primary key default uuid_generate_v4(),
    dispute_id      text references disputes (id) on delete cascade,
    started_at      timestamptz not null default now(),
    finished_at     timestamptz,
    status          text not null default 'running',   -- running | ok | failed
    tool_calls      integer default 0,
    input_tokens    integer,
    output_tokens   integer,
    error           text
);


-- ------------------------------------------------------------- migrations
-- Idempotent fixes for databases created by an earlier version of this file.
-- `create table if not exists` will not alter an existing table, so column
-- changes have to be stated separately.

alter table disputes alter column order_id drop not null;

alter table evidence_packets
    add column if not exists guard jsonb not null default '{}';

alter table disputes
    add column if not exists origin text not null default 'synthetic';

-- Backfill for databases that predate the column. A Stripe object id is 24
-- base62 characters after the prefix; the generator's were 20 hex, so the two
-- are separable after the fact. New rows never rely on this — the webhook
-- records its own provenance.
update disputes
   set origin = 'processor'
 where origin = 'synthetic'
   and processor_dispute_id ~ '^du_[A-Za-z0-9]{24,}$';


-- === REQUIRES PGVECTOR ===
-- Applied tolerantly: if the extension is unavailable the rest of the schema
-- still lands. Only similarity search over past disputes (day 8) needs this.

create extension if not exists vector;

create table if not exists dispute_embeddings (
    dispute_id      text primary key references disputes (id) on delete cascade,
    embedding       vector(1024),
    summary         text
);
