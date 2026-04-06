"""
Legal Corpus Seeder
===================
Seeds the vector store with sample legal provisions for testing.

In production, you would ingest actual statutes, case law, and regulations.
This seeder provides enough representative content to demonstrate the
RAG pipeline working end-to-end.

Coverage areas (aligned with the sample NDA):
  - Non-compete enforceability standards
  - Indemnification provisions
  - Limitation of liability
  - Confidentiality / trade secrets
  - Intellectual property assignment
  - Contract termination
  - Governing law / choice of law
"""

from __future__ import annotations
from .document_loader import LegalDocumentLoader
from .vector_store import LegalVectorStore


# ── Sample legal corpus ───────────────────────────────────────────────
# Each entry simulates a real legal source. In production, these would
# come from actual statute databases, case law APIs, or curated PDFs.

SAMPLE_CORPUS = [
    # ── Non-Compete / Restrictive Covenants ───────────────────────────
    {
        "source_type": "statute",
        "jurisdiction": "Delaware",
        "title": "Delaware Restrictive Covenant Standards",
        "sections": [
            {
                "section_id": "DE-RC-001",
                "heading": "Non-Compete Enforceability in Delaware",
                "text": (
                    "Delaware courts enforce non-compete agreements only when they are "
                    "reasonable in geographic scope, duration, and the activities restricted. "
                    "The employer must demonstrate a legitimate business interest, such as "
                    "protection of trade secrets, confidential information, or customer "
                    "relationships. Courts apply a reasonableness test considering: (1) whether "
                    "the restriction is necessary to protect the employer's legitimate interest; "
                    "(2) whether the restriction is reasonable in time and geographic scope; and "
                    "(3) whether the restriction imposes an undue hardship on the employee. "
                    "Non-competes exceeding two years in duration are presumptively unreasonable. "
                    "Worldwide geographic restrictions are generally unenforceable unless the "
                    "employer demonstrates global competitive interests."
                ),
            },
            {
                "section_id": "DE-RC-002",
                "heading": "Blue Pencil Doctrine in Delaware",
                "text": (
                    "Delaware courts follow the 'blue pencil' doctrine, which allows courts to "
                    "modify overly broad restrictive covenants rather than voiding them entirely. "
                    "Under this doctrine, a court may narrow an unreasonable geographic scope or "
                    "reduce an excessive duration to make the covenant enforceable. However, "
                    "courts will not rewrite the fundamental nature of the restriction. The "
                    "blue pencil doctrine applies only when the covenant is divisible and the "
                    "unreasonable portions can be severed without altering the essential character "
                    "of the agreement."
                ),
            },
            {
                "section_id": "DE-RC-003",
                "heading": "Non-Solicitation vs Non-Compete Distinction",
                "text": (
                    "Delaware courts distinguish between non-compete and non-solicitation "
                    "clauses. Non-solicitation clauses, which restrict only the solicitation of "
                    "specific clients or employees rather than all competitive activity, are "
                    "subject to a less stringent reasonableness standard. A non-solicitation "
                    "clause may be enforceable even when a broader non-compete would not be, "
                    "provided it is limited to clients or employees with whom the restricted "
                    "party had direct contact or material relationships during employment."
                ),
            },
        ],
    },
    # ── Indemnification ───────────────────────────────────────────────
    {
        "source_type": "legal_treatise",
        "jurisdiction": "General",
        "title": "Indemnification Clause Standards in Commercial Contracts",
        "sections": [
            {
                "section_id": "INDEM-001",
                "heading": "Mutual vs One-Sided Indemnification",
                "text": (
                    "Mutual indemnification clauses, where each party indemnifies the other for "
                    "losses caused by its own breach, are considered the market standard in "
                    "arms-length commercial agreements. One-sided indemnification provisions, "
                    "where only one party bears indemnification obligations, are disfavored by "
                    "courts and may be deemed unconscionable in certain circumstances, "
                    "particularly when there is unequal bargaining power. Courts examine whether "
                    "the indemnifying party had meaningful opportunity to negotiate the terms. "
                    "Best practice: both parties should indemnify each other for their respective "
                    "breaches, with carve-outs for specific risk allocations."
                ),
            },
            {
                "section_id": "INDEM-002",
                "heading": "Survival of Indemnification Obligations",
                "text": (
                    "Indemnification obligations typically survive termination of the agreement. "
                    "However, survival without limitation ('shall survive without limitation') "
                    "may be challenged as unreasonable. Market standard survival periods range "
                    "from 12 to 36 months following termination, depending on the nature of "
                    "the obligations. For intellectual property indemnification, longer survival "
                    "periods (up to the statute of limitations) are common. Courts generally "
                    "enforce reasonable survival periods but may strike unlimited survival "
                    "clauses as unconscionable when combined with one-sided obligations."
                ),
            },
        ],
    },
    # ── Limitation of Liability ───────────────────────────────────────
    {
        "source_type": "legal_treatise",
        "jurisdiction": "General",
        "title": "Limitation of Liability in Commercial Agreements",
        "sections": [
            {
                "section_id": "LOL-001",
                "heading": "Enforceability of Liability Caps",
                "text": (
                    "Limitation of liability clauses are generally enforceable in commercial "
                    "contracts between sophisticated parties. However, courts examine whether "
                    "the cap is commercially reasonable relative to the contract value. A "
                    "liability cap of $100 in a contract involving significant business value "
                    "may be deemed unconscionable or illusory. Market standard liability caps "
                    "are typically set at 1x to 2x the total fees paid or payable under the "
                    "agreement. Courts may refuse to enforce liability caps that effectively "
                    "eliminate all meaningful remedy for the non-breaching party."
                ),
            },
            {
                "section_id": "LOL-002",
                "heading": "Consequential Damages Waivers",
                "text": (
                    "Waivers of consequential, indirect, and special damages are common and "
                    "generally enforceable in commercial contracts. However, one-sided waivers "
                    "(where only one party waives its right to consequential damages) are "
                    "scrutinized more closely. The UCC Section 2-719(3) provides that "
                    "limitation of consequential damages for personal injury in consumer goods "
                    "contracts is prima facie unconscionable. In commercial contracts, mutual "
                    "consequential damage waivers are market standard. The waiver must be "
                    "conspicuous — courts may refuse to enforce waivers buried in fine print."
                ),
            },
        ],
    },
    # ── Confidentiality / Trade Secrets ───────────────────────────────
    {
        "source_type": "statute",
        "jurisdiction": "Delaware",
        "title": "Delaware Uniform Trade Secrets Act (6 Del. C. § 2001-2009)",
        "sections": [
            {
                "section_id": "DUTSA-001",
                "heading": "Definition of Trade Secret",
                "text": (
                    "Under the Delaware Uniform Trade Secrets Act (6 Del. C. § 2001), a trade "
                    "secret is defined as information, including a formula, pattern, compilation, "
                    "program, device, method, technique or process, that: (1) derives independent "
                    "economic value, actual or potential, from not being generally known to, and "
                    "not being readily ascertainable by proper means by, other persons who can "
                    "obtain economic value from its disclosure or use; and (2) is the subject "
                    "of efforts that are reasonable under the circumstances to maintain its "
                    "secrecy. Confidentiality agreements support but do not replace the "
                    "requirement for reasonable protective measures."
                ),
            },
            {
                "section_id": "DUTSA-002",
                "heading": "Remedies for Misappropriation",
                "text": (
                    "Under 6 Del. C. § 2002-2004, remedies for trade secret misappropriation "
                    "include: (1) injunctive relief to prevent actual or threatened "
                    "misappropriation; (2) damages for actual loss and unjust enrichment; and "
                    "(3) in cases of willful and malicious misappropriation, exemplary damages "
                    "not exceeding twice the compensatory damages. The statute of limitations "
                    "for trade secret misappropriation is 3 years from the date the "
                    "misappropriation is discovered or should have been discovered."
                ),
            },
        ],
    },
    # ── Intellectual Property Assignment ──────────────────────────────
    {
        "source_type": "legal_treatise",
        "jurisdiction": "General",
        "title": "IP Assignment Clauses in NDAs and Service Agreements",
        "sections": [
            {
                "section_id": "IP-001",
                "heading": "Scope of IP Assignment in NDAs",
                "text": (
                    "Intellectual property assignment clauses in non-disclosure agreements are "
                    "atypical and should be scrutinized carefully. NDAs are designed to protect "
                    "confidential information, not to transfer ownership of intellectual "
                    "property. When IP assignment provisions appear in an NDA, the receiving "
                    "party should be aware that: (1) the scope may be overly broad, capturing "
                    "pre-existing IP or independently developed work; (2) the assignment may "
                    "lack adequate consideration beyond the NDA itself; (3) many jurisdictions "
                    "require IP assignments to be supported by separate consideration. Best "
                    "practice: IP assignment should be addressed in a separate agreement "
                    "(employment agreement, consulting agreement, or IP assignment agreement) "
                    "with clearly defined scope and consideration."
                ),
            },
            {
                "section_id": "IP-002",
                "heading": "Work Made for Hire vs Assignment",
                "text": (
                    "Under 17 U.S.C. § 101, a 'work made for hire' is either: (1) a work "
                    "prepared by an employee within the scope of employment; or (2) a work "
                    "specially ordered or commissioned for use in certain categories, if the "
                    "parties expressly agree in writing. For independent contractors, the "
                    "work-for-hire doctrine applies only to nine enumerated categories. If the "
                    "work does not fall into these categories, an express assignment is required "
                    "to transfer copyright. Broad assignment clauses that capture 'any ideas "
                    "or concepts' developed during the term may be challenged as overly broad "
                    "and potentially unconscionable."
                ),
            },
        ],
    },
    # ── Contract Termination ──────────────────────────────────────────
    {
        "source_type": "legal_treatise",
        "jurisdiction": "General",
        "title": "Termination Provisions in Commercial Agreements",
        "sections": [
            {
                "section_id": "TERM-001",
                "heading": "Unilateral Termination Rights",
                "text": (
                    "Unilateral termination clauses that allow only one party to terminate "
                    "without cause are generally enforceable but may raise fairness concerns. "
                    "Market standard provisions allow either party to terminate for convenience "
                    "with 30 to 90 days' notice. When only one party holds termination rights, "
                    "courts may consider whether adequate consideration supports the arrangement "
                    "and whether the non-terminating party has any meaningful exit mechanism. "
                    "Immediate termination without notice for breach is standard, but best "
                    "practice includes a cure period (typically 15-30 days) for curable breaches."
                ),
            },
            {
                "section_id": "TERM-002",
                "heading": "Survival Clauses After Termination",
                "text": (
                    "Survival clauses specify which obligations continue after termination. "
                    "Standard surviving provisions include: confidentiality obligations (2-5 "
                    "years is market standard), indemnification, limitation of liability, "
                    "governing law, and dispute resolution. Perpetual confidentiality obligations "
                    "may be enforceable for trade secrets but are disfavored for general "
                    "confidential information. A 7-year survival period for confidentiality "
                    "is on the longer end of market practice; 2-3 years is more typical for NDAs."
                ),
            },
        ],
    },
    # ── Governing Law / Choice of Law ─────────────────────────────────
    {
        "source_type": "statute",
        "jurisdiction": "Delaware",
        "title": "Delaware Choice of Law and Forum Selection",
        "sections": [
            {
                "section_id": "GOV-001",
                "heading": "Enforceability of Choice of Law Provisions",
                "text": (
                    "Delaware strongly favors party autonomy in choice of law provisions. "
                    "Under 6 Del. C. § 2708, parties to a contract involving at least $100,000 "
                    "may agree that the contract shall be governed by Delaware law, regardless "
                    "of whether the contract bears any other relationship to Delaware. The "
                    "Delaware Court of Chancery and Superior Court have jurisdiction over such "
                    "disputes. Forum selection clauses designating Delaware courts are generally "
                    "enforceable unless the opposing party demonstrates that enforcement would "
                    "be unreasonable or unjust."
                ),
            },
            {
                "section_id": "GOV-002",
                "heading": "Fee-Shifting Provisions",
                "text": (
                    "Attorney's fee provisions in commercial contracts are enforceable in "
                    "Delaware. Under the 'American Rule,' each party typically bears its own "
                    "legal costs unless a contract or statute provides otherwise. Prevailing "
                    "party fee-shifting clauses are enforceable but should be mutual — one-sided "
                    "fee-shifting (where only one party can recover fees) may be viewed as "
                    "substantively unconscionable, particularly when combined with other "
                    "one-sided provisions."
                ),
            },
        ],
    },
]


def seed_corpus(
    persist_dir: str = "./legal_corpus/vectordb",
    verbose: bool = True,
) -> LegalVectorStore:
    """
    Seed the vector store with sample legal provisions.

    Returns the populated LegalVectorStore instance.
    """
    loader = LegalDocumentLoader(chunk_size=800, chunk_overlap=150)
    store = LegalVectorStore(persist_dir=persist_dir)

    total_chunks = 0

    for corpus_entry in SAMPLE_CORPUS:
        chunks = loader.load_inline(
            sections=corpus_entry["sections"],
            source_type=corpus_entry["source_type"],
            jurisdiction=corpus_entry["jurisdiction"],
            title=corpus_entry["title"],
        )
        added = store.ingest(chunks)
        total_chunks += added

        if verbose:
            print(f"  📚 {corpus_entry['title'][:60]:<60} → {len(chunks)} chunks ({added} new)")

    if verbose:
        print(f"\n  ✅ Corpus ready: {store.count()} total chunks")
        print(f"  📍 Jurisdictions: {', '.join(store.list_jurisdictions())}")

    return store
