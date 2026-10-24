"""Deterministic local ticket tools. No LLM, network, shell, eval or real users."""
import hashlib
import json
from pathlib import Path
import sqlite3

from jsonschema import Draft202012Validator


class ToolError(ValueError):
    def __init__(self, code, message):
        self.code = code
        super().__init__(message)


def text(max_length=500):
    return {"type": "string", "minLength": 1, "maxLength": max_length}


def definition(name, description, properties, required):
    return {"type": "function", "function": {"name": name, "description": description,
            "parameters": {"type": "object", "properties": properties,
                           "required": required, "additionalProperties": False}}}


VERSION = {"type": "integer", "minimum": 1}
STATUSES = ["open", "in_progress", "resolved", "closed"]
TOOLS = [
    definition("lookup_user", "Find a requester by their exact provided email; do not guess missing email.",
               {"email": text(120)}, ["email"]),
    definition("search_tickets", "Search existing tickets; returns at most 100 entries. Do not guess IDs.",
               {"requester_id": text(40), "status": {"type": "string", "enum": STATUSES}, "query": text()}, []),
    definition("get_ticket", "Read a ticket and its current version before any update.",
               {"ticket_id": text(40)}, ["ticket_id"]),
    definition("create_ticket", "Create a ticket using a requester ID obtained from lookup. Writes need trusted client permission.",
               {"title": text(120), "description": text(2000), "requester_id": text(40),
                "priority": {"type": "string", "enum": ["low", "normal", "high"]}},
               ["title", "description", "requester_id", "priority"]),
    definition("assign_ticket", "Assign to A001 (desktop support) or A002 (network support); expected_version must be the current read version.",
               {"ticket_id": text(40), "assignee_id": text(40), "expected_version": VERSION},
               ["ticket_id", "assignee_id", "expected_version"]),
    definition("update_ticket_status", "Change status: open -> in_progress -> resolved -> closed; no skipping. Read latest version first.",
               {"ticket_id": text(40), "status": {"type": "string", "enum": STATUSES}, "expected_version": VERSION},
               ["ticket_id", "status", "expected_version"]),
    definition("search_kb", "Search local knowledge article titles; use the returned article ID, not guessed IDs.",
               {"query": text()}, ["query"]),
    definition("get_kb_article", "Read one local knowledge article by ID.",
               {"article_id": text(40)}, ["article_id"]),
]
WRITE_TOOLS = {"create_ticket", "assign_ticket", "update_ticket_status"}
TRANSITIONS = {"open": "in_progress", "in_progress": "resolved", "resolved": "closed"}


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


class TicketEnvironment:
    """One isolated episode. Caller permission/idempotency keys are not model arguments.

    The whitelist is an application safety boundary, NOT an OS/container sandbox.
    No arbitrary SQL, code, network or filesystem tool is exposed to the model.
    """
    def __init__(self, database=":memory:", *, allow_write=False, max_calls=64):
        if isinstance(max_calls, bool) or not isinstance(max_calls, int) or not 1 <= max_calls <= 1000:
            raise ValueError("max_calls must be an integer in [1, 1000]")
        if not isinstance(allow_write, bool):
            raise ValueError("allow_write must come from an explicit trusted Boolean")
        if database != ":memory:" and Path(database).exists():
            raise ValueError("episode database exists; preserve it rather than resetting")
        self.connection = sqlite3.connect(str(database))
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys=ON")
        self.allow_write, self.max_calls = allow_write, max_calls
        self.audit = []
        self.validators = {tool["function"]["name"]: Draft202012Validator(tool["function"]["parameters"]) for tool in TOOLS}
        self.handlers = {name: getattr(self, "_" + name) for name in self.validators}
        self.connection.executescript("""
            CREATE TABLE users (user_id TEXT PRIMARY KEY, email TEXT UNIQUE NOT NULL, name TEXT NOT NULL);
            CREATE TABLE agents (agent_id TEXT PRIMARY KEY, name TEXT NOT NULL);
            CREATE TABLE tickets (
                ticket_id TEXT PRIMARY KEY, title TEXT NOT NULL, description TEXT NOT NULL,
                requester_id TEXT REFERENCES users(user_id), priority TEXT NOT NULL,
                status TEXT NOT NULL, assignee_id TEXT REFERENCES agents(agent_id), version INTEGER NOT NULL);
            CREATE TABLE articles (article_id TEXT PRIMARY KEY, title TEXT NOT NULL, content TEXT NOT NULL);
            CREATE TABLE idempotency (key TEXT PRIMARY KEY, fingerprint TEXT NOT NULL, result TEXT NOT NULL);
        """)
        with self.connection:
            self.connection.executemany("INSERT INTO users VALUES (?,?,?)", [
                ("U001", "li.ming@example.test", "Li Ming"), ("U002", "wang.lin@example.test", "Wang Lin")])
            self.connection.executemany("INSERT INTO agents VALUES (?,?)", [("A001", "Desktop Support"), ("A002", "Network Support")])
            self.connection.execute("INSERT INTO tickets VALUES (?,?,?,?,?,?,?,?)",
                ("T0001", "VPN connection issue", "VPN fails to connect", "U001", "normal", "open", None, 1))
            self.connection.executemany("INSERT INTO articles VALUES (?,?,?)", [
                ("KB001", "VPN troubleshooting", "Check network access, then contact Network Support if VPN still fails."),
                ("KB002", "Password reset", "Use the approved password reset portal; never disclose your password.")])

    def close(self):
        self.connection.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def snapshot(self):
        return {table: [dict(row) for row in self.connection.execute(f"SELECT * FROM {table} ORDER BY 1")]
                for table in ("users", "agents", "tickets", "articles")}

    def execute(self, name, arguments, *, idempotency_key=None):
        if len(self.audit) >= self.max_calls:
            return {"ok": False, "error": {"code": "call_budget_exceeded", "message": "episode call budget exhausted"}}
        replayed = False
        try:
            if not isinstance(name, str) or name not in self.handlers:
                raise ToolError("unknown_tool", "only registered local tools may run")
            try:
                encoded_args = canonical(arguments)
            except (TypeError, ValueError, OverflowError):
                raise ToolError("invalid_json_value", "arguments must be finite JSON") from None
            if len(encoded_args) > 12000:
                raise ToolError("argument_size_exceeded", "arguments exceed episode input budget")
            errors = list(self.validators[name].iter_errors(arguments))
            if errors:
                raise ToolError("invalid_arguments_schema", errors[0].message[:300])
            if name in WRITE_TOOLS:
                if not self.allow_write:
                    raise ToolError("write_not_authorized", "trusted caller did not permit write actions")
                if not isinstance(idempotency_key, str) or not 1 <= len(idempotency_key) <= 120:
                    raise ToolError("missing_idempotency_key", "trusted executor must supply a bounded write request key")
            fingerprint = hashlib.sha256(canonical({"name": name, "arguments": arguments}).encode()).hexdigest()
            with self.connection:
                previous = self.connection.execute("SELECT * FROM idempotency WHERE key=?", (idempotency_key,)).fetchone() if name in WRITE_TOOLS else None
                if previous:
                    if previous["fingerprint"] != fingerprint:
                        raise ToolError("idempotency_conflict", "same request key cannot name a different action")
                    result, replayed = json.loads(previous["result"]), True
                else:
                    result = self.handlers[name](**arguments)
                    if name in WRITE_TOOLS:
                        self.connection.execute("INSERT INTO idempotency VALUES (?,?,?)", (idempotency_key, fingerprint, canonical(result)))
            response = {"ok": True, "result": result, "replayed": replayed}
        except ToolError as exc:
            response = {"ok": False, "error": {"code": exc.code, "message": str(exc)}}
        # Audit is detached from caller-owned mutable dictionaries.
        try:
            safe_arguments = json.loads(canonical(arguments))
        except (TypeError, ValueError, OverflowError):
            safe_arguments = {"non_json_arguments": str(type(arguments))}
        event = {"index": len(self.audit) + 1, "tool": name if isinstance(name, str) else None,
                 "arguments": safe_arguments, "write_permission": self.allow_write,
                 "idempotency_key": idempotency_key if isinstance(idempotency_key, str) else None,
                 "response": json.loads(canonical(response))}
        self.audit.append(event)
        return response

    def _lookup_user(self, email):
        row = self.connection.execute("SELECT * FROM users WHERE email=?", (email,)).fetchone()
        if not row:
            raise ToolError("user_not_found", "no exact email match; ask for clarification")
        return dict(row)

    def _search_tickets(self, requester_id=None, status=None, query=None):
        conditions, parameters = [], []
        if requester_id is not None:
            conditions.append("requester_id=?")
            parameters.append(requester_id)
        if status is not None:
            conditions.append("status=?")
            parameters.append(status)
        if query is not None:
            conditions.append("instr(lower(title || ' ' || description), lower(?)) > 0")
            parameters.append(query)
        sql = "SELECT * FROM tickets" + (" WHERE " + " AND ".join(conditions) if conditions else "")
        return [dict(row) for row in self.connection.execute(sql + " ORDER BY ticket_id LIMIT 100", parameters)]

    def _get_ticket(self, ticket_id):
        row = self.connection.execute("SELECT * FROM tickets WHERE ticket_id=?", (ticket_id,)).fetchone()
        if not row:
            raise ToolError("ticket_not_found", "unknown ticket ID; do not invent existing tickets")
        return dict(row)

    def _create_ticket(self, title, description, requester_id, priority):
        if not self.connection.execute("SELECT 1 FROM users WHERE user_id=?", (requester_id,)).fetchone():
            raise ToolError("user_not_found", "lookup the requester before creating a ticket")
        count = self.connection.execute("SELECT COUNT(*) FROM tickets").fetchone()[0]
        ticket_id = f"T{count + 1:04d}"
        self.connection.execute("INSERT INTO tickets VALUES (?,?,?,?,?,?,?,?)",
            (ticket_id, title, description, requester_id, priority, "open", None, 1))
        return self._get_ticket(ticket_id)

    def _assign_ticket(self, ticket_id, assignee_id, expected_version):
        row = self._get_ticket(ticket_id)
        if row["version"] != expected_version:
            raise ToolError("version_conflict", "read the latest ticket before retrying with a new request key")
        if row["status"] == "closed":
            raise ToolError("ticket_closed", "closed tickets cannot be reassigned")
        if not self.connection.execute("SELECT 1 FROM agents WHERE agent_id=?", (assignee_id,)).fetchone():
            raise ToolError("assignee_not_found", "unknown local support agent")
        changed = self.connection.execute("UPDATE tickets SET assignee_id=?,version=version+1 WHERE ticket_id=? AND version=?",
                                          (assignee_id, ticket_id, expected_version)).rowcount
        if changed != 1:
            raise ToolError("version_conflict", "concurrent update; read the latest ticket")
        return self._get_ticket(ticket_id)

    def _update_ticket_status(self, ticket_id, status, expected_version):
        row = self._get_ticket(ticket_id)
        if row["version"] != expected_version:
            raise ToolError("version_conflict", "read the latest ticket before retrying with a new request key")
        if TRANSITIONS.get(row["status"]) != status:
            raise ToolError("invalid_status_transition", "only open -> in_progress -> resolved -> closed is allowed")
        changed = self.connection.execute("UPDATE tickets SET status=?,version=version+1 WHERE ticket_id=? AND version=?",
                                          (status, ticket_id, expected_version)).rowcount
        if changed != 1:
            raise ToolError("version_conflict", "concurrent update; read the latest ticket")
        return self._get_ticket(ticket_id)

    def _search_kb(self, query):
        return [dict(row) for row in self.connection.execute(
            "SELECT article_id,title FROM articles WHERE instr(lower(title),lower(?)) > 0 ORDER BY article_id LIMIT 100", (query,))]

    def _get_kb_article(self, article_id):
        row = self.connection.execute("SELECT * FROM articles WHERE article_id=?", (article_id,)).fetchone()
        if not row:
            raise ToolError("article_not_found", "use an ID from the knowledge search result")
        return dict(row)
