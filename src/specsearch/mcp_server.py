"""FastMCP server — exposes search_specs as a tool for Claude Desktop."""
from fastmcp import FastMCP

from specsearch.pipeline import run

mcp = FastMCP(
    "SpecSearch",
    instructions=(
        "Search construction specification PDFs using hybrid BM25 + vector retrieval. "
        "Returns cited answers referencing the exact source file and page number."
    ),
)


@mcp.tool()
def search_specs(query: str) -> str:
    """Search indexed construction specs and return a grounded cited answer.

    Args:
        query: The question to answer from the spec corpus.

    Returns:
        A cited answer with [filename p.N] references to the source documents.
    """
    answer = run(query)
    lines = [answer.text]
    if answer.citations:
        lines.append("\nSources: " + ", ".join(answer.citations))
    return "\n".join(lines)


if __name__ == "__main__":
    mcp.run()
