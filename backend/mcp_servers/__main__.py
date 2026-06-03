import sys


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python -m mcp_servers <server_name>", file=sys.stderr)
        print("Available: codebase", file=sys.stderr)
        sys.exit(2)

    name = sys.argv[1]
    if name == "codebase":
        from mcp_servers.codebase_server import run_stdio
        run_stdio()
    else:
        print(f"Unknown MCP server: {name}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
