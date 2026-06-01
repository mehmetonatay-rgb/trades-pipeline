"""One-time helper — create the Leads and Routes databases (Spec §3.5 / §6).

Run once against a parent page your integration can access, then copy the printed
IDs into .env (NOTION_LEADS_DB_ID / NOTION_ROUTES_DB_ID).

    python -m src.notion_setup --parent-page <PAGE_ID>
"""
from __future__ import annotations

import argparse
import os

from dotenv import load_dotenv
from notion_client import Client


def _select(options: list[str]) -> dict:
    return {"select": {"options": [{"name": o} for o in options]}}


LEADS_SCHEMA = {
    "Name": {"title": {}},
    "Place ID": {"rich_text": {}},
    "Trade": _select([]),
    "Classification": _select(["Service", "Supply", "Uncertain"]),
    "Confidence": {"number": {"format": "number"}},
    "Phone": {"phone_number": {}},
    "Address": {"rich_text": {}},
    "District": _select([]),
    "Neighborhood": {"rich_text": {}},
    "Latitude": {"number": {"format": "number"}},
    "Longitude": {"number": {"format": "number"}},
    "Maps URL": {"url": {}},
    "Rating": {"number": {"format": "number"}},
    "Reviews": {"number": {"format": "number"}},
    "Website": {"url": {}},
    "Status": _select(
        ["New", "Contacted", "Interested", "Scheduled", "Filmed", "Declined"]
    ),
    "Source query": {"rich_text": {}},
    "Notes": {"rich_text": {}},
}

ROUTES_SCHEMA = {
    "Name": {"title": {}},
    "District": _select([]),
    "Stops": {"number": {"format": "number"}},
    "Distance km": {"number": {"format": "number"}},
    "Maps link": {"url": {}},
    "Status": _select(["Planned", "In progress", "Done"]),
}


def _create_db(client: Client, parent_page: str, title: str, schema: dict) -> str:
    db = client.databases.create(
        parent={"type": "page_id", "page_id": parent_page},
        title=[{"type": "text", "text": {"content": title}}],
        properties=schema,
    )
    return db["id"]


def _reshape_db(client: Client, db_id: str, schema: dict, relation_key: str) -> None:
    """Add only the missing properties to an existing DB; never touch its title or
    any column it already has (CRM fields stay as harmless extras). The relation
    property is handled separately, after both DBs are reshaped."""
    db = client.databases.retrieve(database_id=db_id)
    title = "".join(t.get("plain_text", "") for t in db.get("title", []))
    existing = set(db.get("properties", {}).keys())
    to_add = {
        name: spec
        for name, spec in schema.items()
        if name not in existing and name != relation_key and "title" not in spec
    }
    if to_add:
        client.databases.update(database_id=db_id, properties=to_add)
    print(f"  '{title}' ({db_id}): added {len(to_add)} prop(s) "
          f"{sorted(to_add)} | kept {len(existing)} existing")


def reshape_existing() -> None:
    """Reshape the two databases referenced by NOTION_LEADS_DB_ID / NOTION_ROUTES_DB_ID."""
    client = Client(auth=os.environ["NOTION_TOKEN"])
    leads_id = os.environ["NOTION_LEADS_DB_ID"]
    routes_id = os.environ["NOTION_ROUTES_DB_ID"]

    print("Reshaping existing databases:")
    _reshape_db(client, leads_id, LEADS_SCHEMA, relation_key="Route")
    _reshape_db(client, routes_id, ROUTES_SCHEMA, relation_key="Leads")

    # Two-way relation, added once both schemas are in place.
    client.databases.update(
        database_id=leads_id,
        properties={"Route": {"relation": {"database_id": routes_id, "single_property": {}}}},
    )
    client.databases.update(
        database_id=routes_id,
        properties={"Leads": {"relation": {"database_id": leads_id, "single_property": {}}}},
    )
    print("  linked Leads <-> Routes relation. Done.")


def create_fresh(parent_page: str) -> None:
    client = Client(auth=os.environ["NOTION_TOKEN"])
    leads_id = _create_db(client, parent_page, "Trades Leads", LEADS_SCHEMA)
    routes_id = _create_db(client, parent_page, "Trades Routes", ROUTES_SCHEMA)
    client.databases.update(
        database_id=leads_id,
        properties={"Route": {"relation": {"database_id": routes_id, "single_property": {}}}},
    )
    client.databases.update(
        database_id=routes_id,
        properties={"Leads": {"relation": {"database_id": leads_id, "single_property": {}}}},
    )
    print("Add these to .env:")
    print(f"NOTION_LEADS_DB_ID={leads_id}")
    print(f"NOTION_ROUTES_DB_ID={routes_id}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create or reshape the Notion databases.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--parent-page", help="Create two fresh DBs under this page ID")
    group.add_argument("--reshape", action="store_true",
                       help="Add missing columns to the existing NOTION_*_DB_ID databases")
    args = parser.parse_args()

    load_dotenv()
    if args.reshape:
        reshape_existing()
    else:
        create_fresh(args.parent_page)


if __name__ == "__main__":
    main()
