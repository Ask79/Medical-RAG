import sys, os, argparse, re, json
import xml.etree.ElementTree as ET

# Argument parser setup
parser = argparse.ArgumentParser(description="Normalize DailyMed SPL XML to JSONL")
parser.add_argument("--in-dir", required=True, help="Input directory containing SPL XML files")
parser.add_argument("--out", required=True, help="Output file path for newline-delimited JSON")
parser.add_argument("--max-files", type=int, help="Maximum number of files to process", default=None)
parser.add_argument("--verbose", action='store_true', help="Enable verbose output")
args = parser.parse_args()

in_dir = args.in_dir
out_path = args.out
max_files = args.max_files
verbose = args.verbose

# Validate input directory
if not os.path.isdir(in_dir):
    print(f"Error: Input directory '{in_dir}' does not exist or is not a directory.", file=sys.stderr)
    sys.exit(1)

# RECURSIVELY gather XML files (handles many nested folders)
xml_files = []
for root_dir, _, filenames in os.walk(in_dir):
    for fn in filenames:
        if fn.lower().endswith(".xml"):
            xml_files.append(os.path.join(root_dir, fn))

if not xml_files:
    print(f"Error: No XML files found under '{in_dir}' (recursively).", file=sys.stderr)
    sys.exit(1)

xml_files.sort()
if max_files is not None:
    xml_files = xml_files[:max_files]

if verbose:
    print(f"[final] Found {len(xml_files)} XML file(s) under {in_dir}", file=sys.stderr)
    for sample in xml_files[:5]:
        print(f"[final] e.g. {sample}", file=sys.stderr)

# Prepare mapping of known LOINC section names to canonical keys, and a set of known headings
def generate_heading_mappings():
    headings_list = [
        "ABUSE SECTION", "ACCESSORIES", "ADVERSE REACTIONS SECTION", "ALARMS",
        "ANIMAL PHARMACOLOGY & OR TOXICOLOGY SECTION", "ASSEMBLY OR INSTALLATION INSTRUCTIONS",
        "BOXED WARNING SECTION", "CALIBRATION INSTRUCTIONS",
        "CARCINOGENESIS & MUTAGENESIS & IMPAIRMENT OF FERTILITY SECTION",
        "CLINICAL PHARMACOLOGY SECTION", "CLEANING, DISINFECTING, AND STERILIZATION INSTRUCTIONS",
        "CLINICAL STUDIES SECTION", "CLINICAL TRIALS EXPERIENCE SECTION", "COMPATIBLE ACCESSORIES",
        "COMPONENTS", "CONTRAINDICATIONS SECTION", "CONTROLLED SUBSTANCE SECTION",
        "DEPENDENCE SECTION", "DESCRIPTION SECTION", "DIAGRAM OF DEVICE", "DISPOSAL AND WASTE HANDLING",
        "DOSAGE & ADMINISTRATION SECTION", "DOSAGE FORMS & STRENGTHS SECTION",
        "DRUG & OR LABORATORY TEST INTERACTIONS SECTION", "DRUG ABUSE AND DEPENDENCE SECTION",
        "DRUG INTERACTIONS SECTION", "ENVIRONMENTAL WARNING SECTION",
        "FEMALES & MALES OF REPRODUCTIVE POTENTIAL SECTION", "FOOD SAFETY WARNING SECTION",
        "GENERAL PRECAUTIONS SECTION", "GERIATRIC USE SECTION", "GUARANTEED ANALYSIS OF FEED SECTION",
        "HEALTH CARE PROVIDER LETTER SECTION", "HEALTH CLAIM SECTION", "HEPATIC IMPAIRMENT SUBSECTION",
        "HOW SUPPLIED SECTION", "IMMUNOGENICITY", "INACTIVE INGREDIENT SECTION",
        "INDICATIONS & USAGE SECTION", "INFORMATION FOR OWNERS/CAREGIVERS SECTION",
        "INFORMATION FOR PATIENTS SECTION", "INSTRUCTIONS FOR USE SECTION",
        "INTENDED USE OF THE DEVICE", "LABOR & DELIVERY SECTION", "LABORATORY TESTS SECTION",
        "LACTATION SECTION", "MECHANISM OF ACTION SECTION", "MICROBIOLOGY SECTION",
        "NONCLINICAL TOXICOLOGY SECTION", "NONTERATOGENIC EFFECTS SECTION", "NURSING MOTHERS SECTION",
        "OTHER SAFETY INFORMATION", "OVERDOSAGE SECTION",
        "OTC - ACTIVE INGREDIENT SECTION", "OTC - ASK DOCTOR SECTION", "OTC - ASK DOCTOR/PHARMACIST SECTION",
        "OTC - DO NOT USE SECTION", "OTC - KEEP OUT OF REACH OF CHILDREN SECTION",
        "OTC - PREGNANCY OR BREAST FEEDING SECTION", "OTC - PURPOSE SECTION", "OTC - QUESTIONS SECTION",
        "OTC - STOP USE SECTION", "OTC - WHEN USING SECTION", "PACKAGE LABEL.PRINCIPAL DISPLAY PANEL",
        "PATIENT COUNSELING INFORMATION", "PATIENT MEDICATION INFORMATION SECTION",
        "PEDIATRIC USE SECTION", "PHARMACODYNAMICS SECTION", "PHARMACOGENOMICS SECTION",
        "PHARMACOKINETICS SECTION", "POSTMARKETING EXPERIENCE SECTION", "PRECAUTIONS SECTION",
        "PREGNANCY SECTION", "RECENT MAJOR CHANGES SECTION", "REFERENCES SECTION",
        "RESIDUE WARNING SECTION", "REMS ADDRESSED RISK", "REMS ADMINISTRATIVE INFORMATION",
        "REMS APPLICANT REQUIREMENTS", "REMS COMMUNICATION", "REMS ELEMENTS",
        "REMS ELEMENTS TO ASSURE SAFE USE", "REMS GOALS", "REMS IMPLEMENTATION SYSTEM",
        "REMS MATERIAL", "REMS MEDICATION GUIDE", "REMS PARTICIPANT REQUIREMENTS",
        "REMS REQUIREMENTS", "REMS SUMMARY", "REMS TIMETABLE FOR SUBMISSION ASSESSMENTS",
        "RENAL IMPAIRMENT SUBSECTION", "RISKS", "ROUTE, METHOD AND FREQUENCY OF ADMINISTRATION",
        "SAFE HANDLING WARNING SECTION", "SPL INDEXING DATA ELEMENTS SECTION",
        "SPL PRODUCT DATA ELEMENTS SECTION", "SPL MEDGUIDE SECTION",
        "SPL PATIENT PACKAGE INSERT SECTION", "SPL UNCLASSIFIED SECTION",
        "STATEMENT OF IDENTITY SECTION", "STORAGE AND HANDLING SECTION",
        "SUMMARY OF SAFETY AND EFFECTIVENESS", "TERATOGENIC EFFECTS SECTION", "TROUBLESHOOTING",
        "USE IN SPECIFIC POPULATIONS SECTION", "USER SAFETY WARNINGS SECTION",
        "VETERINARY INDICATIONS SECTION", "WARNINGS AND PRECAUTIONS SECTION", "WARNINGS SECTION"
    ]
    mapping = {}
    known_set = set()
    for name in headings_list:
        h = re.sub(r'\s+SECTION$','', name)
        h = re.sub(r'\s+SUBSECTION$','', h).strip()
        low = h.lower()
        t = low.replace('& or', ' and ').replace('&', ' and ')
        if re.search(r'\w/\w', t):
            t = t.replace('/', '_or_')
        else:
            t = t.replace('/', ' ')
        t = re.sub(r'\s+', ' ', t).strip()
        key = re.sub(r'[^\w]', '_', t.replace(' ', '_'))
        key = re.sub(r'_+','_', key).strip('_')
        mapping[low] = key
        known_set.add(low)
    if 'general precautions' in known_set and 'general' not in known_set:
        known_set.add('general')
        mapping['general'] = mapping['general precautions']
    return mapping, known_set

mapping, known_headings = generate_heading_mappings()

# Function to recursively process a section and its subsections
def process_section(sec_elem, sections_dict, unclassified_sections):
    ns = {'hl7': 'urn:hl7-org:v3'}
    code_elem = sec_elem.find('hl7:code', ns)
    sec_key = None
    code_display = None
    code_val = None
    if code_elem is not None:
        code_val = code_elem.get('code')
        code_system_name = code_elem.get('codeSystemName')
        code_display = code_elem.get('displayName')
        if code_system_name == 'LOINC':
            cd_lower = (code_display or "").lower()
            if cd_lower in mapping:
                sec_key = mapping[cd_lower]
            else:
                sec_key = 'unclassified'
        else:
            sec_key = 'unclassified'
    else:
        sec_key = 'unclassified'
    title_elem = sec_elem.find('hl7:title', ns)
    section_title_text = "".join(title_elem.itertext()).strip() if title_elem is not None else None

    # Collect narrative content
    paras = []
    text_elem = sec_elem.find('hl7:text', ns)
    if text_elem is not None:
        if text_elem.text and text_elem.text.strip():
            paras.append(('content', text_elem.text.strip()))
        for child in text_elem:
            tag = child.tag.split('}',1)[1] if '}' in child.tag else child.tag
            tag_lower = tag.lower()
            if tag_lower == 'title':
                subtitle = "".join(child.itertext()).strip()
                if code_display and subtitle.upper() == code_display.upper():
                    pass
                else:
                    paras.append(('heading', subtitle))
            elif tag_lower == 'paragraph':
                p_text = ""
                for node in child.iter():
                    if node is child:
                        if node.text:
                            p_text += node.text
                    else:
                        subtag = node.tag.split('}',1)[1] if '}' in node.tag else node.tag
                        if subtag.lower() == 'br':
                            p_text += "\n"
                        else:
                            if node.text:
                                p_text += node.text
                        if node.tail:
                            p_text += node.tail
                if child.tail:
                    p_text += child.tail
                p_text = p_text.strip()
                if p_text:
                    paras.append(('content', p_text))
            elif tag_lower == 'list':
                for item in child.findall('.//hl7:item', ns):
                    item_text = "".join(item.itertext()).strip()
                    if item_text:
                        paras.append(('content', item_text))
                if child.tail and child.tail.strip():
                    paras.append(('content', child.tail.strip()))
            else:
                other_text = "".join(child.itertext()).strip()
                if other_text:
                    paras.append(('content', other_text))
                if child.tail and child.tail.strip():
                    paras.append(('content', child.tail.strip()))
        if text_elem.tail and text_elem.tail.strip():
            paras.append(('content', text_elem.tail.strip()))

    # Split into parent vs subsections by detecting headings
    parent_content_parts = []
    current_sub_key = None
    current_sub_parts = []
    for ptype, text in paras:
        if ptype == 'heading':
            if current_sub_key is not None:
                sections_dict[current_sub_key] = "\n\n".join(current_sub_parts).strip()
                current_sub_key = None
                current_sub_parts = []
            heading_text = text.strip()
            h_low = heading_text.lower()
            if h_low in mapping:
                sub_key = mapping[h_low]
            else:
                t = heading_text.lower().replace('& or',' and ').replace('&',' and ')
                if re.search(r'\w/\w', t):
                    t = t.replace('/', '_or_')
                else:
                    t = t.replace('/', ' ')
                t = re.sub(r'\s+', ' ', t).strip()
                sub_key = re.sub(r'[^\w]', '_', t.replace(' ', '_'))
                sub_key = re.sub(r'_+','_', sub_key).strip('_')
            current_sub_key = sub_key
            current_sub_parts = []
            continue
        else:
            if current_sub_key is None:
                colon_idx = text.find(':')
                if colon_idx != -1:
                    potential_heading = text[:colon_idx].strip()
                    if potential_heading.lower() in known_headings:
                        heading_text = potential_heading
                        after_colon = text[colon_idx+1:].strip()
                        h_low = heading_text.lower()
                        if h_low in mapping:
                            sub_key = mapping[h_low]
                        else:
                            t = h_low.replace('& or',' and ').replace('&',' and ')
                            if re.search(r'\w/\w', t):
                                t = t.replace('/', '_or_')
                            else:
                                t = t.replace('/', ' ')
                            t = re.sub(r'\s+', ' ', t).strip()
                            sub_key = re.sub(r'[^\w]', '_', t.replace(' ', '_'))
                            sub_key = re.sub(r'_+','_', sub_key).strip('_')
                        current_sub_key = sub_key
                        current_sub_parts = []
                        if after_colon:
                            current_sub_parts.append(after_colon)
                        continue
                parent_content_parts.append(text)
            else:
                colon_idx = text.find(':')
                if colon_idx != -1:
                    potential_heading = text[:colon_idx].strip()
                    if potential_heading.lower() in known_headings:
                        sections_dict[current_sub_key] = "\n\n".join(current_sub_parts).strip()
                        current_sub_parts = []
                        heading_text = potential_heading
                        after_colon = text[colon_idx+1:].strip()
                        h_low = heading_text.lower()
                        if h_low in mapping:
                            sub_key = mapping[h_low]
                        else:
                            t = h_low.replace('& or',' and ').replace('&',' and ')
                            if re.search(r'\w/\w', t):
                                t = t.replace('/', '_or_')
                            else:
                                t = t.replace('/', ' ')
                            t = re.sub(r'\s+', ' ', t).strip()
                            sub_key = re.sub(r'[^\w]', '_', t.replace(' ', '_'))
                            sub_key = re.sub(r'_+','_', sub_key).strip('_')
                        current_sub_key = sub_key
                        current_sub_parts = []
                        if after_colon:
                            current_sub_parts.append(after_colon)
                        continue
                current_sub_parts.append(text)

    if current_sub_key is not None:
        sections_dict[current_sub_key] = "\n\n".join(current_sub_parts).strip()
        current_sub_key = None
        current_sub_parts = []

    parent_content = "\n\n".join(parent_content_parts).strip()
    if sec_key == 'unclassified':
        if section_title_text:
            unclassified_sections[section_title_text] = parent_content
        else:
            key = code_val or "unknown_section"
            unclassified_sections[key] = parent_content
    else:
        sections_dict[sec_key] = parent_content

    # Recursively handle coded nested sections
    for comp in sec_elem.findall('hl7:component', ns):
        child_sec = comp.find('hl7:section', ns)
        if child_sec is not None:
            process_section(child_sec, sections_dict, unclassified_sections)

# Open output file and process each XML
with open(out_path, 'w', encoding='utf-8') as out_file:
    total = len(xml_files)
    for i, filepath in enumerate(xml_files, start=1):
        if verbose:
            print(f"[final] Processing {i}/{total}: {filepath}", file=sys.stderr)
        try:
            tree = ET.parse(filepath)
        except Exception as e:
            print(f"Error parsing XML file {filepath}: {e}", file=sys.stderr)
            continue
        root = tree.getroot()
        ns = {'hl7': 'urn:hl7-org:v3'}

        # Extract metadata fields
        set_id = None
        set_id_elem = root.find('hl7:setId', ns)
        if set_id_elem is not None:
            set_id = set_id_elem.get('root')

        doc_id = None
        doc_id_elem = root.find('hl7:id', ns)
        if doc_id_elem is not None:
            doc_root = doc_id_elem.get('root')
            doc_ext = doc_id_elem.get('extension')
            if doc_ext and doc_ext.endswith('.xml'):
                doc_id = doc_root or doc_ext[:-4]
            else:
                doc_id = doc_root or doc_id_elem.get('extension')

        version_number = None
        ver_elem = root.find('hl7:versionNumber', ns)
        if ver_elem is not None:
            ver_val = ver_elem.get('value')
            if ver_val:
                try:
                    version_number = int(ver_val)
                except:
                    version_number = ver_val

        effective_date = None
        eff_elem = root.find('hl7:effectiveTime', ns)
        if eff_elem is not None:
            effective_date = eff_elem.get('value')

        title = None
        title_elem = root.find('hl7:title', ns)
        if title_elem is not None:
            title = "".join(title_elem.itertext()).strip()

        manufacturer = None
        man_org = root.find('.//hl7:manufacturerOrganization', ns)
        if man_org is not None:
            name_elem = man_org.find('hl7:name', ns)
            if name_elem is not None:
                manufacturer = "".join(name_elem.itertext()).strip()
        if not manufacturer:
            rep_org = root.find('.//hl7:representedOrganization', ns)
            if rep_org is not None:
                name_elem = rep_org.find('hl7:name', ns)
                if name_elem is not None:
                    manufacturer = "".join(name_elem.itertext()).strip()

        # Route and Dosage form
        route = None
        route_raw = None
        route_elem = root.find('.//hl7:routeCode', ns)
        if route_elem is not None:
            route_raw_val = route_elem.get('displayName') or route_elem.get('code')
            if route_raw_val:
                route_raw = route_raw_val.strip()
                route = route_raw.lower()

        dosage_form = None
        form_elem = root.find('.//hl7:formCode', ns)
        if form_elem is not None:
            form_name = form_elem.get('displayName')
            if form_name:
                dosage_form = form_name.strip()

        # Extract and normalize NDCs from text
        xml_text = ET.tostring(root, encoding='unicode', method='text')
        ndc_pattern = re.compile(r'\b\d{4,5}-\d{3,4}-\d{1,2}\b')
        ndc_matches = set(ndc_pattern.findall(xml_text) or [])
        product_ndc_list = []
        package_ndc_list = []
        for ndc in ndc_matches:
            if not re.match(r'^\d+-\d+-\d+$', ndc):
                continue
            p1, p2, p3 = ndc.split('-')
            if len(p1) == 4: p1 = p1.zfill(5)
            if len(p2) == 3: p2 = p2.zfill(4)
            if len(p3) == 1: p3 = p3.zfill(2)
            ndc11 = f"{p1}-{p2}-{p3}"
            if ndc11.endswith('-00'):
                product_ndc_list.append(ndc11)
            else:
                package_ndc_list.append(ndc11)
                base = ndc11.rsplit('-', 1)[0]
                product_ndc_list.append(f"{base}-00")
        product_ndc_list = sorted(set(product_ndc_list))
        package_ndc_list = sorted(set(package_ndc_list))

        # Extract sections recursively
        sections = {}
        unclassified_sections = {}
        structured_body = root.find('.//hl7:structuredBody', ns)
        if structured_body is not None:
            for comp in structured_body.findall('hl7:component', ns):
                sec_elem = comp.find('hl7:section', ns)
                if sec_elem is not None:
                    process_section(sec_elem, sections, unclassified_sections)
        if unclassified_sections:
            sections['unclassified'] = unclassified_sections

        boxed_warning_present = bool('boxed_warning' in sections)

        # Relative path from in_dir for provenance
        provenance_rel = os.path.relpath(filepath, in_dir)

        output_obj = {
            "set_id": set_id,
            "id": doc_id,
            "version_number": version_number,
            "effective_date": effective_date,
            "title": title,
            "route": route,
            "route_raw": route_raw,
            "dosage_form": dosage_form,
            "product_ndc11": product_ndc_list if product_ndc_list else [],
            "package_ndc11": package_ndc_list if package_ndc_list else [],
            "boxed_warning_present": boxed_warning_present,
            "manufacturer": manufacturer,
            "sections": sections,
            "provenance": provenance_rel
        }
        output_obj = {k: v for k, v in output_obj.items() if v is not None}
        out_file.write(json.dumps(output_obj, ensure_ascii=False) + "\n")

if verbose:
    print(f"[final] Processed {len(xml_files)} file(s). Output written to {out_path}", file=sys.stderr)
