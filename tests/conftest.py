from pathlib import Path

import fitz
import pytest

_FIXTURE_DIR = Path(__file__).parent / "fixtures"
_SAMPLE_PDF = _FIXTURE_DIR / "sample_spec.pdf"

_PAGE1 = """\
03 30 00 CAST-IN-PLACE CONCRETE

PART 1 - GENERAL

1.1 SUMMARY
This section specifies cast-in-place concrete including formwork reinforcement and finishing.

1.2 SUBMITTALS
Submit mix designs for each concrete class.
Concrete compressive strength shall be 4000 psi at 28 days.
Portland cement shall conform to ASTM C150 Type I or Type II.

PART 2 - PRODUCTS

2.1 MATERIALS
Cement: ASTM C150 Type I/II. Aggregate: ASTM C33. Water: potable.
Admixtures shall conform to ASTM C494. No calcium chloride admixtures permitted.
"""

_PAGE2 = """\
PART 3 - EXECUTION

3.1 INSTALLATION
Place concrete at ambient temperature between 50 and 90 degrees F.
Cure for minimum 7 days using approved curing compound or wet burlap.

03 45 00 PRECAST CONCRETE

1.1 GENERAL
Precast concrete panels shall meet ACI 318 requirements.
All precast units shall be manufactured in a PCI-certified plant.
"""


@pytest.fixture(scope="session")
def sample_spec_pdf() -> Path:
    """Two-page synthetic spec PDF with two CSI sections."""
    _FIXTURE_DIR.mkdir(exist_ok=True)
    if not _SAMPLE_PDF.exists():
        doc = fitz.open()
        for text in (_PAGE1, _PAGE2):
            doc.new_page().insert_text((50, 50), text, fontsize=11)
        doc.save(str(_SAMPLE_PDF))
        doc.close()
    return _SAMPLE_PDF
