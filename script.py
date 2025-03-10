import os
import glob
import shutil
import logging
import re
import requests
import xml.etree.ElementTree as ET
import pandas as pd
import gradio as gr
import torch
import copy
import json
from subprocess import run
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
from bs4 import BeautifulSoup

# =======================
# CONFIGURATION
# =======================
PROJECT_HOME = '/Users/peter/ai-stuff/citation-checker/data/'  # adjust as needed
GrobidUrl = "http://localhost:8070/api/processFulltextDocument"
CrossrefMailto = "mailto=Peter.tamas@wur.nl"  # your email

NLI_MODELS_DEFAULT = {
    "BlackBeanie DeBERTa-V3-Large": ("BlackBeenie/nli-deberta-v3-large", 512),
    "cross-encoder/nli-deberta-v3-base": ("cross-encoder/nli-deberta-v3-base", 1024),
    "cointegrated/rubert-base-cased-nli-threeway": ("cointegrated/rubert-base-cased-nli-threeway", 512),
    "amoux/scibert_nli_squad": ("amoux/scibert_nli_squad", 512)
}

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# =======================
# GLOBAL PROJECT FOLDER CREATION
# =======================
def sanitize_filename(filename):
    """Replace invalid filename characters with underscores."""
    return "".join(c if c.isalnum() or c in (' ', '.', '_') else '_' for c in filename).replace(' ', '_')

def create_project_folder(project_name):
    top_folder = os.path.join(PROJECT_HOME, sanitize_filename(project_name))
    subfolders = {
        "citing_pdf": os.path.join(top_folder, "citing_article", "pdf"),
        "citing_tei": os.path.join(top_folder, "citing_article", "tei"),
        "cited_pdf": os.path.join(top_folder, "cited_articles", "pdf"),
        "cited_tei": os.path.join(top_folder, "cited_articles", "tei"),
        "tables": os.path.join(top_folder, "tables")
    }
    for path in subfolders.values():
        os.makedirs(path, exist_ok=True)
    return subfolders

project_subfolders = None  # Global state


def get_existing_projects():
    if not os.path.exists(PROJECT_HOME):
        return []
    return [d for d in os.listdir(PROJECT_HOME) if os.path.isdir(os.path.join(PROJECT_HOME, d))]

def normalize_text(text):
    text = re.sub(r'\.\s*\.\s*\.', '...', text)
    text = text.replace('…', '...')
    return re.sub(r'\s+', ' ', text).strip()

# =======================
# GROBID + TEI EXTRACTION (unchanged)
# =======================
def mark_as_retrieved(doi, project_name):
    csv_path = os.path.join(project_subfolders["tables"], "cited.csv")
    if not os.path.exists(csv_path):
        logging.error(f"cited.csv not found at {csv_path}")
        return
    df = pd.read_csv(csv_path)
    # Update rows where DOI matches
    df.loc[df["DOI"] == doi, "retrieved"] = "Yes"
    df.to_csv(csv_path, index=False)
    logging.info(f"Updated retrieved status for DOI {doi} in cited.csv")

def process_pdf_with_grobid(pdf_path, output_dir=None, doi=None, project_name=None):
    try:
        with open(pdf_path, 'rb') as f:
            files = {'input': f}
            params = {
                'consolidateHeader': '1',
                'consolidateCitations': '1',
                'includeRawAffiliations': '1',
                'segmentSentences': '1'
            }
            response = requests.post(GrobidUrl, files=files, data=params)
            if response.status_code == 200:
                tei_xml = response.text
                logging.info("Grobid processing succeeded.")
                if output_dir:
                    os.makedirs(output_dir, exist_ok=True)
                    tei_filename = os.path.splitext(os.path.basename(pdf_path))[0] + ".tei.xml"
                    tei_output_path = os.path.join(output_dir, tei_filename)
                    with open(tei_output_path, "w", encoding="utf-8") as f_out:
                        f_out.write(tei_xml)
                # If DOI and project name are provided, update the cited.csv
                if doi and project_name:
                    mark_as_retrieved(doi, project_name)
                return tei_xml

                return tei_xml
            else:
                logging.error(f"Grobid failed with status {response.status_code}")
    except Exception as e:
        logging.error(f"Grobid processing exception: {e}")
    return None

def process_pdf_interface(pdf_file, project_name):
    global project_subfolders
    if not pdf_file:
        return "No file provided.", ""
    if not project_name:
        project_name = os.path.splitext(os.path.basename(pdf_file))[0]
    project_subfolders = create_project_folder(project_name)
    pdf_basename = os.path.basename(pdf_file)
    citing_pdf_path = os.path.join(project_subfolders["citing_pdf"], pdf_basename)
    shutil.copy(pdf_file, citing_pdf_path)
    tei_xml = process_pdf_with_grobid(citing_pdf_path, output_dir=project_subfolders["citing_tei"])
    return tei_xml if tei_xml else "Failed to extract TEI.", project_name, project_subfolders


def load_existing_tei(project_name):
    project_path = os.path.join(PROJECT_HOME, project_name, "citing", "tei")
    tei_files = glob.glob(os.path.join(project_path, "*.tei.xml"))
    if tei_files:
        with open(tei_files[0], "r", encoding="utf-8") as f:
            return f.read()
    return "No TEI XML found in project."

# =======================
# CITATION CHUNK EXTRACTION (unchanged)
# =======================
def extract_bibr_chunks(tei_xml):
    soup = BeautifulSoup(tei_xml, "xml")
    body = soup.find("body")
    if not body:
        return []
    items = []
    sent_id = 0
    for s_tag in body.find_all("s"):
        refs = s_tag.find_all("ref", {"type": "bibr"})
        if not refs:
            continue
        s_tag_copy = BeautifulSoup(str(s_tag), "xml")
        for ref in s_tag_copy.find_all("ref", {"type": "bibr"}):
            ref.decompose()
        clean_text = s_tag_copy.get_text(" ", strip=True)
        for ref in refs:
            target = ref.get("target", "")
            items.append({
                "sentence_id": f"sent_{sent_id}",
                "bibr_target": target,
                "original_text": clean_text,
                "edited_text": clean_text
            })
        sent_id += 1
    return items

def get_citation_chunks_df(tei_xml):
    chunks = extract_bibr_chunks(tei_xml)
    df = pd.DataFrame(chunks)
    if not df.empty:
        df["Include"] = "Yes"
        df = df[["Include", "bibr_target", "original_text", "edited_text"]]
    return df

# =======================
# CSV HANDLING (updated)
# =======================

def combined_csv_callback(csv_file, tei_xml, project_name):
    """
    1) Load corrected CSV and build the reference table.
    2) Immediately run consolidation (DOI lookups, create/update 'cited.csv').
    """
    corrected_df, ref_table_df, ref_table_state = load_csv_and_build_table(csv_file, tei_xml)
    if corrected_df.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), "", "No rows to consolidate."
    
    consolidated_df, consolidated_html, progress_message = consolidate_and_show_progress(
        corrected_df, tei_xml, project_name
    )
    return (
        corrected_df,           # corrected_chunks_state
        ref_table_df,           # editable_reference_table
        ref_table_state,        # reference_table_state
        consolidated_html,      # consolidated HTML preview
        progress_message        # consolidation progress logs
    )

def prepare_citation_chunks_csv(tei_xml, project_name):
    """
    Processes the TEI XML to extract citing sentence chunks and writes a CSV file with:
      - original_text: text of sentence chunk
      - sentence_id: unique identifier for each citing chunk
      - bibr_targets: comma separated list of #bnn values
    The CSV is written to both the temporary folder and the project's tables folder,
    and is named as: citing.csv.
    """
    df = get_citation_chunks_df(tei_xml)
    if df.empty:
        return None
    # Reset index to use as sentence_id, rename columns accordingly.
    df = df.reset_index().rename(columns={"index": "sentence_id", "original_text": "original_text"})
    df = df.rename(columns={"bibr_target": "bibr_targets"})
    # Group by the original_text to ensure uniqueness and merge bibr_targets.
    df_grouped = df.groupby("original_text", as_index=False).agg({
        "sentence_id": "first",
        "bibr_targets": lambda x: ", ".join(x),
        "Include": "first"
    })

    # Add the new column 'cleaned_text' and set its default value equal to 'original_text'
    df_grouped["cleaned_text"] = df_grouped["original_text"]
    # Ensure temporary directory exists.
    temp_dir = "temp"
    os.makedirs(temp_dir, exist_ok=True)
    csv_filename = f"citing.csv"
    csv_path = os.path.join(temp_dir, csv_filename)
    df_grouped.to_csv(csv_path, index=False)
    # Also write CSV to the 'tables' folder if available.
    if project_subfolders and "tables" in project_subfolders:
        tables_csv_path = os.path.join(project_subfolders["tables"], csv_filename)
        df_grouped.to_csv(tables_csv_path, index=False)
    return csv_path

def load_corrected_csv(csv_file):
    df = pd.read_csv(csv_file)
    return df

def load_csv_and_build_table(csv_file, tei):
    if not csv_file:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    df = load_corrected_csv(csv_file)
    table_df, table_state_df = build_editable_reference_table(df, tei)
    return df, table_df, table_state_df

def update_citing_csv(persistent_csv_path, submitted_csv_path):
    """
    Load the persistent citing CSV and the user-submitted CSV.
    If the persistent CSV does not have the 'cleaned_text' column, create it.
    Update the persistent CSV so that for each matching sentence_id, the 'cleaned_text'
    column is set to the value from the submitted CSV's "original_text" column.
    Rows without a matching sentence_id in the submitted CSV will have an empty 'cleaned_text' value.
    Duplicate rows (same sentence_id) are preserved.
    """
    persistent_df = pd.read_csv(persistent_csv_path)
    submitted_df = pd.read_csv(submitted_csv_path)
    
    # Create the 'cleaned_text' column if it does not exist.
    if "cleaned_text" not in persistent_df.columns:
        persistent_df["cleaned_text"] = ""
    
    # Update each row based on sentence_id matches.
    for idx, row in persistent_df.iterrows():
        sid = row["sentence_id"]
        matching = submitted_df[submitted_df["sentence_id"] == sid]
        if not matching.empty:
            persistent_df.at[idx, "cleaned_text"] = matching.iloc[0]["original_text"]
        else:
            persistent_df.at[idx, "cleaned_text"] = ""
    
    persistent_df.to_csv(persistent_csv_path, index=False)
    logging.info(f"Persistent citing CSV updated at {persistent_csv_path}")
    return persistent_df

# ---- UPDATE load_cited_articles to format bib_info as "author (year) title" ----
def load_cited_articles():
    if not project_subfolders:
        return []
    cited_tei_folder = project_subfolders.get("cited_tei", "")
    articles = []
    for file in glob.glob(os.path.join(cited_tei_folder, "*.tei.xml")):
        with open(file, "r", encoding="utf-8") as f:
            tei_xml = f.read()
        soup = BeautifulSoup(tei_xml, "xml")
        # Extract author, title, and date in the desired order.
        author_tag = soup.find("author")
        title_tag = soup.find("title")
        date_tag = soup.find("date")
        author = author_tag.get_text(strip=True) if author_tag else "Unknown Author"
        title = title_tag.get_text(strip=True) if title_tag else "No Title"
        year = date_tag.get("when") if date_tag and date_tag.get("when") else "Unknown Year"
        bib_info = f"{author} ({year}) {title}"
        doi = os.path.splitext(os.path.basename(file))[0]
        articles.append({"doi": doi, "bib_info": bib_info, "tei_xml": tei_xml})
    return articles

def load_citing_csv(project_name):
    """
    Load the citing CSV file for the given project name from the project's tables folder.
    The file is expected to be named 'citing.csv'.
    Returns a pandas DataFrame. If not found, returns an empty DataFrame.
    """
    if not project_subfolders or "tables" not in project_subfolders:
        logging.error("Project subfolders not set or 'tables' folder missing.")
        return pd.DataFrame()
    
    sanitized_project_name = sanitize_filename(project_name)
    csv_filename = f"citing.csv"
    citing_csv_path = os.path.join(project_subfolders["tables"], csv_filename)
    
    if os.path.exists(citing_csv_path):
        return pd.read_csv(citing_csv_path)
    else:
        logging.error(f"Could not find {citing_csv_path}")
        return pd.DataFrame()

# =======================
# CONSOLIDATION USING TEI REFERENCES
# =======================
def extractReferencesFromTei(teiContent):
    """
    Extract references from TEI XML content using ElementTree.
    Also extract xml:id attribute for each biblStruct.
    """
    references = []
    try:
        logging.debug("Extracting references from TEI content.")
        root = ET.fromstring(teiContent)
        for bibl in root.findall('.//{http://www.tei-c.org/ns/1.0}biblStruct'):
            xml_id = bibl.attrib.get('{http://www.w3.org/XML/1998/namespace}id', 'unknown_id')
            existingDoi = bibl.find('.//{http://www.tei-c.org/ns/1.0}idno[@type="DOI"]')
            if existingDoi is not None:
                logging.info(f"DOI already present: {existingDoi.text}")
                continue
            reference = {
                'title': bibl.find('.//{http://www.tei-c.org/ns/1.0}title').text if bibl.find('.//{http://www.tei-c.org/ns/1.0}title') is not None else '',
                'author': ', '.join([
                    author.find('.//{http://www.tei-c.org/ns/1.0}surname').text
                    for author in bibl.findall('.//{http://www.tei-c.org/ns/1.0}author')
                    if author.find('.//{http://www.tei-c.org/ns/1.0}surname') is not None
                ]),
                'year': bibl.find('.//{http://www.tei-c.org/ns/1.0}date').get('when') if bibl.find('.//{http://www.tei-c.org/ns/1.0}date') is not None else '',
                'xml_id': xml_id
            }
            references.append({'element': bibl, 'reference': reference})
        logging.info(f"Extracted {len(references)} references from TEI content.")
    except Exception as e:
        logging.error(f"Error extracting references from TEI: {e}")
    return references

def findDoi(reference):
    """
    Search for a DOI online using CrossRef based on reference details.
    """
    query = "Unknown Query"
    try:
        author = reference.get("author", "")
        title = reference.get("title", "")
        year = reference.get("year", "")
        query = f"{author} {title} {year}"
        logging.debug(f"Searching for DOI with query: {query}")
        url = f"https://api.crossref.org/works?query.bibliographic={query}&mailto={CrossrefMailto}"
        response = requests.get(url)
        if response.status_code == 200:
            items = response.json().get("message", {}).get("items", [])
            if items:
                doi = items[0].get("DOI")
                logging.info(f"DOI found: {doi} for query: {query}")
                return doi
        logging.info(f"No DOI found for query: {query}")
    except Exception as e:
        logging.error(f"Error fetching DOI for query: {query}, Error: {e}")
    return None

def consolidate_using_references(corrected_df, tei_xml):
    """
    For each row in the corrected CSV (where "Include" == "Yes"),
    use the bibr_target field to look up matching bibliographic reference from TEI,
    then query CrossRef to obtain a DOI.
    Returns a new DataFrame with consolidated DOI entries.
    """
    references = extractReferencesFromTei(tei_xml)
    ref_dict = {}
    for ref in references:
        ref_id = ref['reference'].get('xml_id', '')
        ref_dict[ref_id] = ref['reference']
    doi_cache = {}
    consolidated = []
    for idx, row in corrected_df.iterrows():
        if row["Include"] != "Yes":
            continue
        citing_text = row.get("edited_text", "")
        raw = row.get("bibr_target")
        if raw is None or pd.isna(raw) or (isinstance(raw, str) and raw.strip() == ""):
            logging.info(f"Row {idx} skipped due to missing bibr_target.")
            continue
        targets = [t.strip() for t in str(raw).split(",") if t.strip()]
        doi_list = []
        for t in targets:
            t_id = t.lstrip("#")
            if t_id in doi_cache:
                doi = doi_cache[t_id]
            else:
                if t_id in ref_dict:
                    doi = findDoi(ref_dict[t_id])
                    doi_cache[t_id] = doi if doi is not None else "Not found"
                else:
                    doi = "Reference not found"
            doi_list.append(f"DOI: {doi}")
        doi_str = "; ".join(doi_list)
        entry = f"{citing_text} | {doi_str}"
        consolidated.append({
            "Citation Chunk": citing_text,
            "Consolidated Entry": entry
        })
    return pd.DataFrame(consolidated)

def create_cited_references_csv(tei_xml, project_name):
    """
    Create a CSV file named "cited.csv" using information extracted from the TEI XML.
    The CSV will have columns: bibr_target, author, title, year, DOI, sanitized_DOI.
    """
    references = extractReferencesFromTei(tei_xml)
    rows = []
    for ref in references:
        r = ref['reference']
        xml_id = r.get("xml_id", "")
        author = r.get("author", "")
        title = r.get("title", "")
        year = r.get("year", "")
        doi = findDoi(r)
        doi = doi if doi is not None else ""
        sanitized_doi = doi.replace("/", "_") if doi else ""
        rows.append({
            "bibr_target": xml_id,
            "author": author,
            "title": title,
            "year": year,
            "DOI": doi,
            "sanitized_DOI": sanitized_doi,
            "retrieved": "No"
        })
    df = pd.DataFrame(rows)
    output_dir = project_subfolders.get("tables") if project_subfolders and "tables" in project_subfolders else "temp"
    os.makedirs(output_dir, exist_ok=True)
    sanitized_project_name = sanitize_filename(project_name)
    csv_filename = f"cited.csv"
    csv_path = os.path.join(output_dir, csv_filename)
    df.to_csv(csv_path, index=False)
    return csv_path


def consolidate_and_show_progress(corrected_df, tei_xml, project_name):
    """
    Run consolidation automatically upon CSV upload.
    Returns the consolidated DataFrame, a styled HTML version, and a progress message.
    """
    progress_msgs = []
    progress_msgs.append("Starting consolidation...")
    consolidated_df = consolidate_using_references(corrected_df, tei_xml)
    progress_msgs.append("DOI queries complete.")
    progress_msgs.append("Consolidation finished.")
    # Silently create the cited references CSV.
    cited_csv_path = create_cited_references_csv(tei_xml, project_name)
    progress_msgs.append(f"Cited CSV created: {cited_csv_path}")
    progress_str = "\n".join(progress_msgs)
    return consolidated_df, get_styled_dataframe_html(consolidated_df), progress_str
    
    
    progress_str = "\n".join(progress_msgs)
    return consolidated_df, get_styled_dataframe_html(consolidated_df), progress_str


def build_editable_reference_table(corrected_df, tei_xml):
    """
    Build an editable table with three columns:
      - "Reference ID" (e.g., "#b30")
      - "Bib Info" (e.g., "Author (Year), Title")
      - "DOI" (auto-filled if found; editable)
    Returns the table DataFrame twice.
    """
    references = extractReferencesFromTei(tei_xml)
    ref_dict = {}
    for ref in references:
        ref_id = ref['reference'].get('xml_id', '')
        author = (ref['reference'].get("author") or "").strip()
        year = (ref['reference'].get("year") or "").strip()
        title = (ref['reference'].get("title") or "").strip()
        bib_info = f"{author} ({year}), {title}".strip()
        query = f"{author}, {title}, {year}".strip()
        ref_dict[ref_id] = {"bib_info": bib_info, "query": query, "ref": ref['reference']}
    
    unique_refs = set()
    for idx, row in corrected_df.iterrows():
        if row.get("Include") != "Yes":
            continue
        raw = row.get("bibr_target", "")
        if raw is None or pd.isna(raw) or str(raw).strip() == "":
            continue
        targets = [t.strip().lstrip("#") for t in str(raw).split(",") if t.strip()]
        unique_refs.update(targets)
    
    rows = []
    for ref_id in unique_refs:
        if ref_id in ref_dict:
            ref_display = f"#{ref_id}"
            bib_info = ref_dict[ref_id]["bib_info"]
            doi = findDoi(ref_dict[ref_id]["ref"])
            if doi is None:
                doi = ""
        else:
            ref_display = f"#{ref_id}"
            bib_info = "Reference not found"
            doi = ""
        rows.append({
            "Reference ID": ref_display,
            "Bib Info": bib_info,
            "DOI": doi
        })
    df = pd.DataFrame(rows)
    # Add a default Status column for later retrieval processing.
    df["Status"] = "No PDF yet"
    return df, df

# =======================
# RETRIEVAL FUNCTIONS
# =======================
def get_api_email():
    return CrossrefMailto.split("mailto=")[1]

def run_subprocess(cmd):
    return run(cmd, capture_output=True, text=True)

def download_pdf_via_pypaperbot(doi):
    cited_pdf = project_subfolders["cited_pdf"]
    os.makedirs(cited_pdf, exist_ok=True)
    temp_dir = os.path.join(cited_pdf, 'temp')
    os.makedirs(temp_dir, exist_ok=True)
    cmd = [
        "PyPaperBot",
        f"--doi={doi}",
        f"--dwn-dir={temp_dir}",
    ]
    result = run_subprocess(cmd)
    if result.returncode == 0:
        downloaded_files = glob.glob(os.path.join(temp_dir, "*.pdf"))
        if downloaded_files:
            new_filename = doi.replace("/", "_") + ".pdf"
            new_filepath = os.path.join(cited_pdf, new_filename)
            os.rename(downloaded_files[0], new_filepath)
            shutil.rmtree(temp_dir)
            return new_filepath
    shutil.rmtree(temp_dir)
    return None

def retrieve_pdf(doi):
    email = get_api_email()
    api_url = f"https://api.unpaywall.org/v2/{doi}?email={email}"
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        r = requests.get(api_url, headers=headers, timeout=20)
        if r.status_code == 200:
            data = r.json()
            best_loc = data.get("best_oa_location")
            if best_loc and best_loc.get("url_for_pdf"):
                pdf_url = best_loc["url_for_pdf"]
                pdf_resp = requests.get(pdf_url, headers=headers, timeout=20)
                if pdf_resp.status_code == 200 and 'application/pdf' in pdf_resp.headers.get("Content-Type", ""):
                    return pdf_url, pdf_resp.content
            return None, "No open-access PDF found"
        else:
            return None, f"Unpaywall error: {r.status_code}"
    except Exception as e:
        return None, f"Error: {e}"

def retrieve_and_process_article_with_progress(doi, progress=gr.Progress()):
    progress(0, desc=f"Starting processing for DOI {doi}")
    pdf_url, pdf_content = retrieve_pdf(doi)
    progress(0.3, desc="Checked Unpaywall")
    cited_pdf = project_subfolders["cited_pdf"]
    cited_tei = project_subfolders["cited_tei"]
    if pdf_url is not None:
        progress(0.4, desc="Open-access PDF found via Unpaywall")
        new_filename = doi.replace("/", "_") + ".pdf"
        new_filepath = os.path.join(cited_pdf, new_filename)
        with open(new_filepath, "wb") as f:
            f.write(pdf_content)
        progress(0.6, desc="PDF saved. Processing with Grobid...")
        tei = process_pdf_with_grobid(new_filepath, output_dir=cited_tei)
        if tei and "Failed" not in tei:
            progress(1, desc="Grobid conversion successful.")
            return "successful", "Grobid conversion successful."
        else:
            progress(1, desc="Grobid conversion failed.")
            return "failed", "Grobid conversion failed."
    else:
        progress(0.4, desc="No open-access PDF via Unpaywall; trying PyPaperBot...")
        pdf_path = download_pdf_via_pypaperbot(doi)
        if pdf_path:
            progress(0.7, desc="PDF downloaded via PyPaperBot. Processing with Grobid...")
            tei = process_pdf_with_grobid(pdf_path, output_dir=cited_tei)
            if tei and "Failed" not in tei:
                progress(1, desc="Grobid conversion successful.")
                return "successful", "Grobid conversion successful."
            else:
                progress(1, desc="Grobid conversion failed.")
                return "failed", "Grobid conversion failed."
        else:
            progress(1, desc="PyPaperBot download failed.")
            return "failed", "PyPaperBot download failed."

def build_retrieval_dashboard_html(updated_df):
    html = ["<table border='1' style='width:100%; border-collapse: collapse;'>"]
    html.append("<tr><th>Bib Info</th><th>Status</th></tr>")
    for idx, row in updated_df.iterrows():
        bib = row.get("Bib Info", "")
        status = row.get("Status", "in queue")
        html.append(f"<tr><td>{bib}</td><td>{status}</td></tr>")
    html.append("</table>")
    return "\n".join(html)

def start_downloads_stream(ref_table_df, tei_xml, project_name, progress=gr.Progress()):
    cited_csv_path = create_cited_references_csv(tei_xml, project_name)

    updated_df = ref_table_df.copy()
    total = len(updated_df)
    cumulative_log = ""
    for idx, row in updated_df.iterrows():
        doi_cell = row.get("DOI", "")
        doi = doi_cell.split("DOI:")[-1].strip() if "DOI:" in doi_cell else doi_cell.strip()
        bib_info = row.get("Bib Info", "")
        if doi and doi not in ["Not found", "Error"]:
            status, prog_message = retrieve_and_process_article_with_progress(doi, progress=progress)
            updated_df.at[idx, "Status"] = status
            cumulative_log += f"Row {idx+1}/{total} ({bib_info}): {prog_message}\n"
        else:
            updated_df.at[idx, "Status"] = "No PDF yet"
            cumulative_log += f"Row {idx+1}/{total} ({bib_info}): No valid DOI provided.\n"
        dashboard_html = build_retrieval_dashboard_html(updated_df)
        yield dashboard_html, cumulative_log

# =======================
# MANUAL UPLOAD FOR MISSING CITED ARTICLES
# =======================
def process_manual_pdf(pdf_file, doi):
    if not pdf_file:
        return "failed", "No file provided for manual upload."
    cited_pdf = project_subfolders["cited_pdf"]
    cited_tei = project_subfolders["cited_tei"]
    new_filename = doi.replace("/", "_") + ".pdf"
    new_filepath = os.path.join(cited_pdf, new_filename)
    shutil.copy(pdf_file, new_filepath)
    tei = process_pdf_with_grobid(new_filepath, output_dir=cited_tei)
    if tei and "Failed" not in tei:
        return "successful", "Grobid conversion successful for manual upload."
    else:
        return "failed", "Grobid conversion failed for manual upload."

def get_missing_options(ref_table_df):
    if "Status" not in ref_table_df.columns:
        ref_table_df["Status"] = "Manual input needed"
    missing_df = ref_table_df[ref_table_df["Status"] == "Manual input needed"]
    options = []
    mapping = {}
    for idx, row in missing_df.iterrows():
        bib = row.get("Bib Info", "")
        doi = row.get("DOI", "").strip()
        option = f"{bib} (DOI: {doi})"
        options.append(option)
        mapping[option] = doi
    return options, json.dumps(mapping)

def manual_upload_process(selected_option, pdf_file, mapping_json):
    mapping = json.loads(mapping_json)
    doi = mapping.get(selected_option, "")
    return process_manual_pdf(pdf_file, doi)

# =======================
# NEW: SELECTION TAB FUNCTIONS
# =======================

def get_selectable_citing_sentences_from_csv(project_subfolders):
    if project_subfolders is None:
        logging.error("project_subfolders state is None!")
        return []

    csv_path = os.path.join(project_subfolders["tables"], "citing.csv")
    if not os.path.exists(csv_path):
        logging.error(f"citing.csv not found at {csv_path}")
        return []

    df = pd.read_csv(csv_path)
    cited_tei_path = project_subfolders["cited_tei"]
    existing_ref_ids = {os.path.splitext(os.path.basename(p))[0] for p in glob.glob(f"{cited_tei_path}/*.tei.xml")}

    def has_cited_match(bibr_targets):
        targets = [t.strip().lstrip("#") for t in str(bibr_targets).split(",")]
        return any(t in existing_ref_ids for t in targets)

    df_filtered = df[df["bibr_targets"].apply(has_cited_match)]
    choices = []
    for idx, row in df_filtered.iterrows():
        text = row.get("cleaned_text") or row.get("original_text", "")
        if pd.isna(text) or not text.strip():
            continue
        payload = {"row_idx": idx, "text": text.strip()}
        choices.append(json.dumps(payload))
    return choices

def get_references_for_sentence(selected_json, project_name):
    if not selected_json:
        return []  # Return an empty list immediately if input is None or empty
    
    data = json.loads(selected_json)
    row_idx = data["row_idx"]
    df = load_citing_csv(project_name)
    
    if row_idx >= len(df):
        logging.error(f"Row index {row_idx} is out of bounds for DataFrame length {len(df)}.")
        return []

    row = df.iloc[row_idx]
    raw_targets = row.get("bibr_target") or row.get("bibr_targets", "")
    if not raw_targets:
        return []

    ref_ids = [t.strip().lstrip("#") for t in str(raw_targets).split(",") if t.strip()]
    cited_articles = load_cited_articles()
    choices = []

    for art in cited_articles:
        if art.get("doi", "") in ref_ids:
            payload = {
                "doi": art.get("doi", ""),
                "bib_info": art.get("bib_info", ""),
                "tei_xml": art.get("tei_xml", "")
            }
            choices.append(json.dumps(payload))
    return choices

def confirm_citation(selected_sentence_json, selected_reference_json, citation_index):
    if not selected_sentence_json or not selected_reference_json:
        logging.error("Missing selection data for citing sentence or cited reference.")
        return "", "", citation_index
    sentence_data = json.loads(selected_sentence_json)
    ref_data = json.loads(selected_reference_json)

    citing_sentence = sentence_data.get("text", "")
    tei_xml = ref_data.get("tei_xml", "")
    new_index = citation_index + 1
    return citing_sentence, tei_xml, new_index


# =======================
# NLI CITATION CHECKING (unchanged)
# =======================
def load_nli_model_results(model_name):
    if model_name in NLI_MODELS_DEFAULT:
        model_path, max_tokens = NLI_MODELS_DEFAULT[model_name]
    else:
        model_path, max_tokens = model_name, 512
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)
    nli_pipeline = pipeline("text-classification", model=model, tokenizer=tokenizer, device=device)
    return nli_pipeline, tokenizer, max_tokens

def nli_candidates_all_results(model_name, citing_sentence, cited_xml):
    soup = BeautifulSoup(cited_xml, "xml")
    for ref in soup.find_all("ref"):
        ref.decompose()
    sentences = [
        s.get_text(" ", strip=True)
        for s in soup.find_all(lambda tag: tag.name=="s" or tag.name.endswith(":s"))
        if len(s.get_text(" ", strip=True).split()) >= 3
    ]
    if not sentences:
        sentences = [
            sent for sent in re.split(r'(?<=[.!?])\s+', cited_xml.strip())
            if len(sent.split()) >= 3
        ]
    nli_pipeline, tokenizer, max_tokens = load_nli_model_results(model_name)
    all_results = []
    for window_len in [1,2,3]:
        for i in range(len(sentences) - window_len + 1):
            window = sentences[i:i+window_len]
            if any(len(w.split()) < 3 for w in window):
                continue
            window_text = " ".join(window)
            norm_window = normalize_text(window_text)
            input_text = f"{citing_sentence} [SEP] {norm_window}"
            try:
                preds = nli_pipeline(input_text, truncation=True, max_length=max_tokens)
            except Exception:
                continue
            entailments = [p for p in preds if "entail" in p["label"].lower()]
            score = max([p["score"] for p in entailments], default=0.0)
            if score > 0.0:
                all_results.append((window_text, score))
    return sorted(all_results, key=lambda x: x[1], reverse=True)

def nli_candidates_top5_results(model_name, citing_sentence, cited_xml, window_types=[1,2,3]):
    all_ = nli_candidates_all_results(model_name, citing_sentence, cited_xml)
    return all_[:5]

def citation_checker(citing_sentence, tei_xml, citation_id, model_name, window_options):
    soup = BeautifulSoup(tei_xml, "xml")
    body = soup.find("body")
    if not body:
        return "Error: <body> not found", gr.update(choices=[], value=None)
    extracted_context = str(body)
    w_map = {"1 Sentence": 1, "2 Sentences": 2, "3 Sentences": 3}
    w_types = [w_map[o] for o in window_options]
    top5 = nli_candidates_top5_results(model_name, citing_sentence, str(body), w_types)
    if not top5:
        return extracted_context, gr.update(choices=[], value=None)
    radio_list = []
    for i, (ctx, sc) in enumerate(top5, 1):
        label = f"Candidate {i}:\n{ctx}\n(Confidence: {sc:.4f})"
        radio_list.append(label)
    return extracted_context, gr.update(choices=radio_list, value=None)

cases_global = []

def add_case(citing_sentence, tei_xml, citation_id, model_name, window_options, correct_candidate, cases):
    soup = BeautifulSoup(tei_xml, "xml")
    body = soup.find("body")
    extracted_context = str(body) if body else "No <body> found"
    new_case = {
        "citing_sentence": citing_sentence,
        "tei_xml": tei_xml,
        "citation_id": citation_id,
        "window_options": window_options,
        "correct_candidate": correct_candidate,
        "extracted_context": extracted_context
    }
    cases.append(new_case)
    return cases, f"Case added. Total cases: {len(cases)}."

def find_rank_conf(case, model_name):
    lines = case["correct_candidate"].splitlines()
    text_only = lines[1].strip() if len(lines) >= 2 else case["correct_candidate"].strip()
    all_ = nli_candidates_all_results(model_name, case["citing_sentence"], case["extracted_context"])
    for idx, (ctx, score) in enumerate(all_, 1):
        if ctx.strip() == text_only:
            if idx <= 5:
                return f"<span style='color: green;'>{idx} (Conf: {score:.4f})</span>"
            else:
                return f"{idx} (Conf: {score:.4f})"
    return "<span style='color: red;'>N/A</span>"

def run_comparison(custom_links, cases):
    lines = [l.strip() for l in custom_links.splitlines() if l.strip()]
    all_models = dict(NLI_MODELS_DEFAULT)
    for link in lines:
        if link not in all_models:
            all_models[link] = (link, 512)
    col_headers = [f"Case {i+1} ({c['citation_id']})" for i, c in enumerate(cases)]
    html = ["<table border='1' style='border-collapse: collapse;'>"]
    html.append("<tr><th>Model</th>")
    for col in col_headers:
        html.append(f"<th>{col}</th>")
    html.append("</tr>")
    for model in all_models:
        html.append(f"<tr><td>{model}</td>")
        for c in cases:
            rank_html = find_rank_conf(c, model)
            html.append(f"<td style='padding:5px;'>{rank_html}</td>")
        html.append("</tr>")
    html.append("</table>")
    return "\n".join(html)

def get_styled_dataframe_html(df):
    df = df.rename(columns={
        "Include": "Include? (Y/N)",
        "original_text": "Original Sentence (modify as appropriate)"
    })
    html_table = df.to_html(escape=False, index=False)
    style = """
    <style>
        table { width: 100%; font-family: Arial, sans-serif; }
        th, td { padding: 8px; border: 1px solid #ddd; text-align: left; vertical-align: top; white-space: normal; word-wrap: break-word; }
    </style>
    """
    return style + html_table

# =======================
# GRADIO INTERFACE
# =======================
with gr.Blocks(css=".gradio-container { font-family: sans-serif; }") as demo:
    gr.Markdown("## Citation Processing & Benchmarking (Auto-process on PDF Drop)")
    
    # Remove the separate project name textbox.
    # Instead, the project name will be derived from the PDF filename.
    
    project_name_state = gr.State("")
    project_subfolders_state = gr.State()

    with gr.Tabs():
        # Tab 1: PDF & Citation Chunks
        with gr.TabItem("PDF & Citation Chunks"):
            gr.Markdown(
                "Drop a citing article PDF to create a project folder with the following structure:\n"
                "  - citing_article/pdf & citing_article/tei\n"
                "  - cited_articles/pdf & cited_articles/tei\n"
                "  - tables\n\n"
                "The citing PDF will be processed by Grobid to extract TEI XML and a CSV will be generated for editing."
            )
            # Only PDF input is now needed.
            pdf_input = gr.File(label="Drop citing PDF here", type="filepath", file_count="single")
            tei_output = gr.Textbox(label="Citing TEI XML (hidden)", lines=8, visible=False)
            
            # Process PDF input; project name is derived inside process_pdf_interface.
            pdf_input.change(
                fn=process_pdf_interface,
                inputs=[pdf_input],  # Only pass the PDF file now.
                outputs=[tei_output, project_name_state, project_subfolders_state]
            )

            # Generate CSV from TEI.
            csv_download = gr.File(label="Download CSV of citing sentences to correct", interactive=False)
            tei_output.change(
                fn=prepare_citation_chunks_csv,
                inputs=[tei_output, project_name_state],  # project_name_state is used here.
                outputs=csv_download
            )
            
            # States and reference table for the corrected CSV.
            corrected_csv = gr.File(label="Upload corrected CSV (do not rename)", type="filepath")
            corrected_chunks_state = gr.State(pd.DataFrame())
            reference_table_state = gr.State(pd.DataFrame())
            # Make sure to name the outputs correctly (adjust variable names if needed).
            editable_reference_table = gr.Dataframe(
                label="Reference Table: Correct DOIs by double clicking on the cell",
                headers=["Reference ID", "Bib Info", "DOI"],
                interactive=True
            )
            consoldated_html_display = gr.HTML(label="Consolidated Results")
            consolidation_progress_box = gr.Textbox(label="Consolidation Progress", lines=4, interactive=False)
            
            # Use combined callback for corrected CSV.
            corrected_csv.change(
                fn=combined_csv_callback,
                inputs=[corrected_csv, tei_output, project_name_state],
                outputs=[
                    corrected_chunks_state,
                    editable_reference_table,
                    reference_table_state,
                    consoldated_html_display,
                    consolidation_progress_box
                ]
            )
            
            start_downloads_button = gr.Button("Start Downloads")
            progress_log_box = gr.Textbox(label="Progress Log", interactive=False, lines=6)
            # Use project_name_state instead of a non-existent project_name_input.
            start_downloads_button.click(
                fn=start_downloads_stream,
                inputs=[reference_table_state, tei_output, project_name_state],
                outputs=[gr.HTML(), progress_log_box],

            )

        
        # Tab 2: Upload Missing Cited
        with gr.TabItem("Upload Missing Cited"):
            gr.Markdown(
                "Select a missing article from the dropdown below, then drop the correct PDF beside it and click 'Process Manual Upload'."
            )
            missing_dropdown = gr.Dropdown(label="Missing Articles", choices=[])
            missing_mapping = gr.Textbox(label="Mapping", visible=False)
            refresh_missing_button = gr.Button("Refresh Missing Options")
            refresh_missing_button.click(
                fn=get_missing_options,
                inputs=[reference_table_state],
                outputs=[missing_dropdown, missing_mapping]
            )
            pdf_upload = gr.File(label="Drop PDF here", type="filepath", file_count="single")
            process_manual_button = gr.Button("Process Manual Upload")
            manual_result = gr.Textbox(label="Manual Upload Result", interactive=False)
            process_manual_button.click(
                fn=manual_upload_process,
                inputs=[missing_dropdown, pdf_upload, missing_mapping],
                outputs=manual_result
            )
        
        # NEW Tab 2.5: Selection
        with gr.TabItem("Select citing"):
            gr.Markdown("Select a citing sentence from your saved CSV and then select a corresponding cited article based on its bibliographic info.")
            
            # Display the stored project name read-only
            project_name_display = gr.Textbox(label="Project Name", interactive=False)
            project_name_display.change(fn=lambda x: x, inputs=[project_name_state], outputs=[project_name_display])
            
            # Radio for citing sentences from the saved citing CSV.
            citing_sentence_radio = gr.Radio(label="Select Citing Sentence", choices=[])
            refresh_sentences_btn = gr.Button("Refresh Citing Sentences")
            refresh_sentences_btn.click(
                fn=get_selectable_citing_sentences_from_csv,
                inputs=[project_subfolders_state],
                outputs=citing_sentence_radio
            )
            
            # 2. Radio for cited articles (displaying author (year) title) corresponding to the chosen citing sentence.
            cited_article_radio = gr.Radio(label="Select Cited Article", choices=[])
            refresh_articles_btn = gr.Button("Refresh Articles for Selected Sentence")
            refresh_articles_btn.click(
                fn=get_references_for_sentence,
                inputs=[citing_sentence_radio, project_name_state],
                outputs=cited_article_radio
            )
            
            # 3. Confirm button to finalize selection.
            confirm_btn = gr.Button("Confirm Selection")
            # Hidden fields for downstream NLI processing.
            citing_sentence_output = gr.Textbox(label="Citing Sentence (for NLI)", visible=False)
            tei_xml_output = gr.Textbox(label="Cited TEI XML (for NLI)", visible=False)
            citation_id_output = gr.Textbox(label="Citation ID", visible=False)
            # A citation index stored as a State.
            citation_index_state = gr.State(0)
            
            confirm_btn.click(
                fn=confirm_citation,
                inputs=[citing_sentence_radio, cited_article_radio, citation_index_state],
                outputs=[citing_sentence_output, tei_xml_output, citation_id_output]
            )
            

        # Tab 3: NLI Checking
        with gr.TabItem("NLI Checking"):
            gr.Markdown(
                "Enter a citing sentence, citation ID, and the TEI XML of a cited article.\n"
                "Select an NLI model and window types, then choose the best candidate.\n"
                "These fields can also be updated from the Selection tab."
            )
            citing_sentence_input = gr.Textbox(label="Citing Sentence")
            citation_id_input = gr.Textbox(label="Citation ID")
            tei_xml_input = gr.Textbox(label="Cited TEI XML", lines=8)
            # Pre-populate with selection outputs if available.
            model_dropdown = gr.Dropdown(
                label="Select NLI Model",
                choices=list(NLI_MODELS_DEFAULT.keys()),
                value="BlackBeanie DeBERTa-V3-Large"
            )
            window_checkbox = gr.CheckboxGroup(
                label="Window Types",
                choices=["1 Sentence", "2 Sentences", "3 Sentences"],
                value=["1 Sentence", "2 Sentences", "3 Sentences"]
            )
            run_button = gr.Button("Run Citation Check")
            extracted_context_box = gr.Textbox(label="Extracted Context", lines=5)
            candidate_radio = gr.Radio(label="Select Correct Candidate", choices=[], type="value", interactive=True)
            run_button.click(
                fn=citation_checker,
                inputs=[citing_sentence_input, tei_xml_input, citation_id_input, model_dropdown, window_checkbox],
                outputs=[extracted_context_box, candidate_radio]
            )
            gr.Markdown("Add this result as a benchmark case:")
            add_case_button = gr.Button("Add Case")
            case_status = gr.Textbox(label="Case Status", lines=2)
            cases_state = gr.State([])
            add_case_button.click(
                fn=add_case,
                inputs=[citing_sentence_input, tei_xml_input, citation_id_input, model_dropdown, window_checkbox, candidate_radio, cases_state],
                outputs=[cases_state, case_status]
            )
            gr.Markdown("Run comparison across all stored cases:")
            custom_model_links = gr.Textbox(label="Additional Model Links", lines=4)
            compare_button = gr.Button("Compare")
            compare_html = gr.HTML(label="Comparison Table")
            compare_button.click(
                fn=run_comparison,
                inputs=[custom_model_links, cases_state],
                outputs=compare_html
            )
            citing_sentence_output.change(
                fn=lambda x: x,
                inputs=[citing_sentence_output],
                outputs=[citing_sentence_input]
            )

            tei_xml_output.change(
                fn=lambda x: x,
                inputs=[tei_xml_output],
                outputs=[tei_xml_input]
            )

            citation_id_output.change(
                fn=lambda x: x,
                inputs=[citation_id_output],
                outputs=[citation_id_input]
            )

    
    demo.launch()
