"""
CLI to manage dashboard users.

Usage:
    python manage_users.py add    <email> ["Full Name"]
    python manage_users.py list
    python manage_users.py remove <email>
    python manage_users.py reset  <email>
"""
from __future__ import annotations

import argparse
import getpass
import sys

import auth
import db


def cmd_add(email: str, full_name: str | None) -> int:
    if db.get_user(email) is not None:
        print(f"ERROR: user {email!r} already exists. Use `reset` to change password.")
        return 1
    pw = _prompt_password_twice()
    db.add_user(email, auth.hash_password(pw), full_name)
    print(f"Added user {email}.")
    return 0


def cmd_list() -> int:
    users = db.list_users()
    if not users:
        print("(no users)")
        return 0
    for u in users:
        name = u["full_name"] or ""
        print(f"{u['email']:<40s} {name:<30s} {u['created_at']}")
    return 0


def cmd_remove(email: str) -> int:
    n = db.delete_user(email)
    if n == 0:
        print(f"No user named {email!r}.")
        return 1
    print(f"Removed user {email}.")
    return 0


def cmd_reset(email: str) -> int:
    if db.get_user(email) is None:
        print(f"No user named {email!r}.")
        return 1
    pw = _prompt_password_twice()
    db.update_password(email, auth.hash_password(pw))
    print(f"Password updated for {email}.")
    return 0


def _prompt_password_twice() -> str:
    while True:
        pw = getpass.getpass("Password: ")
        if not pw:
            print("Password cannot be empty.")
            continue
        confirm = getpass.getpass("Confirm:  ")
        if pw != confirm:
            print("Passwords do not match. Try again.")
            continue
        return pw


def main() -> int:
    parser = argparse.ArgumentParser(description="Dashboard user management")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_add = sub.add_parser("add", help="add a user")
    p_add.add_argument("email")
    p_add.add_argument("full_name", nargs="?", default=None)

    sub.add_parser("list", help="list users")

    p_rm = sub.add_parser("remove", help="remove a user")
    p_rm.add_argument("email")

    p_reset = sub.add_parser("reset", help="reset a user's password")
    p_reset.add_argument("email")

    args = parser.parse_args()
    db.initialize()

    if args.cmd == "add":
        return cmd_add(args.email, args.full_name)
    if args.cmd == "list":
        return cmd_list()
    if args.cmd == "remove":
        return cmd_remove(args.email)
    if args.cmd == "reset":
        return cmd_reset(args.email)
    return 2


if __name__ == "__main__":
    sys.exit(main())
