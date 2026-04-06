"""
Generate a sample NDA PDF for testing the Intake Agent.

This creates a realistic-looking NDA with multiple clause types,
including some intentionally risky clauses to test risk detection.
"""

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY


def create_sample_nda(output_path: str = "sample_docs/sample_nda.pdf"):
    doc = SimpleDocTemplate(output_path, pagesize=letter,
                            topMargin=1*inch, bottomMargin=1*inch,
                            leftMargin=1.2*inch, rightMargin=1.2*inch)

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("Title2", parent=styles["Title"], fontSize=16,
                                  spaceAfter=20, alignment=TA_CENTER)
    heading_style = ParagraphStyle("Heading", parent=styles["Heading2"], fontSize=12,
                                    spaceBefore=18, spaceAfter=8, bold=True)
    body_style = ParagraphStyle("Body", parent=styles["Normal"], fontSize=10,
                                 leading=14, alignment=TA_JUSTIFY, spaceAfter=6)
    preamble_style = ParagraphStyle("Preamble", parent=body_style, fontSize=10,
                                     spaceAfter=12)

    elements = []

    # ── Title ──
    elements.append(Paragraph("NON-DISCLOSURE AGREEMENT", title_style))
    elements.append(Spacer(1, 12))

    # ── Preamble ──
    elements.append(Paragraph(
        "This Non-Disclosure Agreement (\"Agreement\") is entered into as of January 15, 2025 "
        "(the \"Effective Date\") by and between Acme Technologies Inc., a Delaware corporation "
        "with its principal office at 100 Innovation Drive, San Francisco, CA 94105 "
        "(\"Disclosing Party\"), and Beta Solutions LLC, a California limited liability company "
        "with its principal office at 200 Market Street, Los Angeles, CA 90012 "
        "(\"Receiving Party\"). The Disclosing Party and Receiving Party are collectively "
        "referred to as the \"Parties\" and individually as a \"Party.\"",
        preamble_style
    ))
    elements.append(Paragraph(
        "WHEREAS, the Disclosing Party possesses certain confidential and proprietary "
        "information relating to its technology, business operations, and strategic plans; and "
        "WHEREAS, the Receiving Party desires to receive such information for the purpose of "
        "evaluating a potential business partnership (the \"Purpose\"); NOW, THEREFORE, in "
        "consideration of the mutual covenants contained herein, the Parties agree as follows:",
        preamble_style
    ))

    # ── 1. Definitions ──
    elements.append(Paragraph("1. DEFINITIONS", heading_style))
    elements.append(Paragraph(
        "\"Confidential Information\" means any and all non-public information, whether "
        "written, oral, electronic, visual, or in any other form, disclosed by the Disclosing "
        "Party to the Receiving Party, including but not limited to: (a) trade secrets, "
        "inventions, patents, patent applications, copyrights, and know-how; (b) business "
        "plans, financial data, customer lists, vendor relationships, pricing strategies, and "
        "marketing plans; (c) software, source code, algorithms, databases, technical "
        "specifications, and system architectures; (d) any information marked as "
        "\"Confidential,\" \"Proprietary,\" or with similar designation; and (e) any "
        "information that a reasonable person would understand to be confidential given the "
        "nature of the information and circumstances of disclosure.",
        body_style
    ))

    # ── 2. Obligations ──
    elements.append(Paragraph("2. OBLIGATIONS OF THE RECEIVING PARTY", heading_style))
    elements.append(Paragraph(
        "The Receiving Party agrees to: (a) hold all Confidential Information in strict "
        "confidence using at least the same degree of care it uses to protect its own "
        "confidential information, but in no event less than reasonable care; (b) not disclose "
        "Confidential Information to any third party without the prior written consent of the "
        "Disclosing Party; (c) limit access to Confidential Information to its employees, "
        "contractors, and advisors who have a need to know and who are bound by confidentiality "
        "obligations at least as restrictive as those contained herein; (d) not use Confidential "
        "Information for any purpose other than the Purpose; and (e) promptly notify the "
        "Disclosing Party upon discovery of any unauthorized use or disclosure.",
        body_style
    ))

    # ── 3. Non-Compete (intentionally one-sided / risky) ──
    elements.append(Paragraph("3. NON-COMPETE AND NON-SOLICITATION", heading_style))
    elements.append(Paragraph(
        "During the term of this Agreement and for a period of five (5) years following its "
        "termination or expiration, the Receiving Party shall not, directly or indirectly: "
        "(a) engage in, own, manage, operate, finance, or participate in any business that "
        "competes with the Disclosing Party anywhere in the world; (b) solicit, recruit, or "
        "hire any employee, contractor, or consultant of the Disclosing Party; or (c) solicit, "
        "divert, or take away any client, customer, or business opportunity of the Disclosing "
        "Party. The Receiving Party acknowledges that these restrictions are reasonable and "
        "necessary to protect the Disclosing Party's legitimate business interests.",
        body_style
    ))

    # ── 4. Intellectual Property (broad assignment) ──
    elements.append(Paragraph("4. INTELLECTUAL PROPERTY", heading_style))
    elements.append(Paragraph(
        "Any ideas, concepts, inventions, discoveries, improvements, or works of authorship "
        "(collectively, \"Developments\") that the Receiving Party conceives, creates, or "
        "reduces to practice, whether alone or jointly with others, during the term of this "
        "Agreement and that relate in any way to the Disclosing Party's business or the "
        "Confidential Information shall be the sole and exclusive property of the Disclosing "
        "Party. The Receiving Party hereby irrevocably assigns to the Disclosing Party all "
        "right, title, and interest in and to such Developments, including all intellectual "
        "property rights therein. The Receiving Party agrees to execute any documents and take "
        "any actions reasonably necessary to perfect the Disclosing Party's ownership.",
        body_style
    ))

    # ── 5. Indemnification (one-sided) ──
    elements.append(Paragraph("5. INDEMNIFICATION", heading_style))
    elements.append(Paragraph(
        "The Receiving Party shall indemnify, defend, and hold harmless the Disclosing Party "
        "and its officers, directors, employees, agents, successors, and assigns from and "
        "against any and all claims, damages, losses, costs, and expenses (including reasonable "
        "attorneys' fees) arising out of or in connection with: (a) any breach of this "
        "Agreement by the Receiving Party; (b) any unauthorized use or disclosure of "
        "Confidential Information; or (c) any violation of applicable law by the Receiving "
        "Party in connection with this Agreement. This indemnification obligation shall "
        "survive the termination or expiration of this Agreement without limitation.",
        body_style
    ))

    # ── 6. Limitation of Liability ──
    elements.append(Paragraph("6. LIMITATION OF LIABILITY", heading_style))
    elements.append(Paragraph(
        "IN NO EVENT SHALL THE DISCLOSING PARTY BE LIABLE TO THE RECEIVING PARTY FOR ANY "
        "INDIRECT, INCIDENTAL, SPECIAL, CONSEQUENTIAL, OR PUNITIVE DAMAGES, REGARDLESS OF "
        "THE CAUSE OF ACTION OR THE THEORY OF LIABILITY, EVEN IF THE DISCLOSING PARTY HAS "
        "BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGES. THE DISCLOSING PARTY'S TOTAL "
        "AGGREGATE LIABILITY UNDER THIS AGREEMENT SHALL NOT EXCEED ONE HUNDRED DOLLARS "
        "($100.00). This limitation shall not apply to the Receiving Party's indemnification "
        "obligations under Section 5.",
        body_style
    ))

    # ── 7. Term and Termination ──
    elements.append(Paragraph("7. TERM AND TERMINATION", heading_style))
    elements.append(Paragraph(
        "This Agreement shall commence on the Effective Date and remain in effect for a period "
        "of three (3) years, unless earlier terminated by the Disclosing Party upon thirty (30) "
        "days' written notice. The Disclosing Party may terminate this Agreement immediately "
        "and without notice if the Receiving Party breaches any provision hereof. Upon "
        "termination, the Receiving Party shall promptly return or destroy all Confidential "
        "Information and certify in writing that it has done so. The confidentiality obligations "
        "under this Agreement shall survive termination for a period of seven (7) years.",
        body_style
    ))

    # ── 8. Governing Law ──
    elements.append(Paragraph("8. GOVERNING LAW AND DISPUTE RESOLUTION", heading_style))
    elements.append(Paragraph(
        "This Agreement shall be governed by and construed in accordance with the laws of the "
        "State of Delaware, without regard to its conflict of laws principles. Any dispute "
        "arising out of or relating to this Agreement shall be resolved exclusively in the "
        "state or federal courts located in Wilmington, Delaware. The Receiving Party "
        "irrevocably consents to the personal jurisdiction of such courts and waives any "
        "objection to venue. The prevailing party in any litigation shall be entitled to "
        "recover its reasonable attorneys' fees and costs.",
        body_style
    ))

    # ── 9. Miscellaneous ──
    elements.append(Paragraph("9. MISCELLANEOUS", heading_style))
    elements.append(Paragraph(
        "This Agreement constitutes the entire agreement between the Parties with respect to "
        "the subject matter hereof and supersedes all prior negotiations, representations, and "
        "agreements. This Agreement may be amended only by a written instrument signed by both "
        "Parties. If any provision of this Agreement is held to be invalid or unenforceable, "
        "the remaining provisions shall continue in full force and effect. The failure of either "
        "Party to enforce any right under this Agreement shall not constitute a waiver of such "
        "right. This Agreement may not be assigned by the Receiving Party without the prior "
        "written consent of the Disclosing Party. This Agreement may be executed in counterparts, "
        "each of which shall be deemed an original.",
        body_style
    ))

    # ── Signature Block ──
    elements.append(Spacer(1, 30))
    elements.append(Paragraph("IN WITNESS WHEREOF, the Parties have executed this Agreement "
                               "as of the Effective Date.", body_style))
    elements.append(Spacer(1, 24))
    elements.append(Paragraph("DISCLOSING PARTY: Acme Technologies Inc.", body_style))
    elements.append(Paragraph("By: _________________________", body_style))
    elements.append(Paragraph("Name: John Smith, CEO", body_style))
    elements.append(Spacer(1, 18))
    elements.append(Paragraph("RECEIVING PARTY: Beta Solutions LLC", body_style))
    elements.append(Paragraph("By: _________________________", body_style))
    elements.append(Paragraph("Name: Jane Doe, Managing Partner", body_style))

    doc.build(elements)
    print(f"✅ Sample NDA created: {output_path}")


if __name__ == "__main__":
    create_sample_nda()
