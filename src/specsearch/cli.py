from pathlib import Path

import typer

from specsearch import config
from specsearch.index import build_es_index, build_vector_index
from specsearch.ingest import ingest_directory
from specsearch.pipeline import run

app = typer.Typer(help="SpecSearch — hybrid RAG over construction-spec PDFs.")


@app.command()
def index(
    directory: Path = typer.Argument(
        config.RAW_DIR, help="Directory containing PDF files to ingest."
    ),
) -> None:
    """Ingest PDFs and build both ES (BM25) and Chroma (vector) indexes."""
    typer.echo(f"Ingesting PDFs from {directory} ...")
    chunks = ingest_directory(directory)
    pdf_count = len(list(directory.glob("*.pdf")))
    typer.echo(f"  {len(chunks)} chunks from {pdf_count} PDF(s)")

    typer.echo("Building Elasticsearch index ...")
    build_es_index(chunks)
    typer.echo(f"  {len(chunks)} docs indexed")

    typer.echo("Building Chroma vector index ...")
    build_vector_index(chunks)
    typer.echo(f"  {len(chunks)} embeddings stored")

    typer.echo("Done.")


@app.command()
def query(
    question: str = typer.Argument(..., help="Question to answer from the spec corpus."),
    explain: bool = typer.Option(False, "--explain", help="Show per-stage details."),
) -> None:
    """Query the indexed specs and return a grounded cited answer."""
    answer = run(question, explain=explain)

    typer.echo("\nANSWER\n------")
    typer.echo(answer.text)

    if answer.citations:
        typer.echo("\nCITATIONS\n---------")
        for c in answer.citations:
            typer.echo(f"  {c}")

    if explain:
        typer.echo("\nTIMINGS\n-------")
        for stage, secs in answer.timings.items():
            typer.echo(f"  {stage:<20} {secs:.3f}s")

        if answer.rerank_scores:
            typer.echo(f"\nTOP-{len(answer.rerank_scores)} CHUNKS (rerank score)\n" + "-" * 35)
            for score, chunk in answer.rerank_scores:
                typer.echo(
                    f"  [{score:+.3f}] {chunk.source_file} p.{chunk.page}"
                    f" — {chunk.section[:60]}"
                )

        typer.echo(
            f"\n{len(answer.candidates)} candidates → {len(answer.rerank_scores)} returned"
        )


@app.command()
def eval(
    no_timings: bool = typer.Option(False, "--no-timings", help="Skip latency benchmark."),
) -> None:
    """Run Recall@10 / MRR / nDCG@10 ablation + latency benchmark → README RESULTS."""
    from specsearch.eval import run_eval

    run_eval(with_timings=not no_timings)
