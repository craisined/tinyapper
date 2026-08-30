# This file is basically fully AI generated, frick regex

import html
import re

def heading_fix(match):
    line = match.group(0).strip()
    leading_equals = len(re.match(r"^(= )+", line).group(0)) // 2
    level = min(max(leading_equals, 1), 3)
    clean_title = line.strip("= ").strip(" =").strip()
    if clean_title == "":
        return ""
    return f"{level * "= "}{clean_title}{level * " ="}"

def clean_tokens(text):

    text = re.sub(r"^\s*=+.*$", heading_fix, text, flags=re.MULTILINE)
    text = re.sub(r"<eos>|<\|endoftext\|>", "", text)
    text = text.replace("<unk>", '<span class="unk-token">[unk]</span>')
    text = text.replace(" @-@ ", "-")
    text = text.replace("@-@", "-")
    text = text.replace(" @,@ ", ",")
    text = text.replace(" @.@ ", ".")
    text = re.sub(r"\b(\w+)\s+('s|'t|'re|'ve|'m|'ll|'d)\b", r"\1\2", text)
    text = re.sub(r"\s+([,\.\?\!\:\;\)\>\]])", r"\1", text)
    text = re.sub(r"([\(\<\[])\s+", r"\1", text)
    text = re.sub(r'"\s+(.*?)\s+"', r'"\1"', text)

    return text

def wikitext_to_html(text):
    text = clean_tokens(text)
    text = html.escape(text)
    text = re.sub(
        r"^\s*=\s*=\s*=\s*(.*?)\s*=\s*=\s*=\s*$",
        r"<h4>\1</h4><hr>",
        text,
        flags=re.MULTILINE,
    )
    text = re.sub(
        r"^\s*=\s*=\s*(.*?)\s*=\s*=\s*$",
        r"<h3>\1</h3><hr>",
        text,
        flags=re.MULTILINE,
    )
    text = re.sub(
        r"^\s*=\s*(.*?)\s*=\s*$",
        r"<h2>\1</h2><hr>",
        text,
        flags=re.MULTILINE,
    )

    raw_blocks = re.split(r"\n\s*\n", text)

    html_blocks = []
    for block in raw_blocks:
        block = block.strip()
        if not block:
            continue
        if re.match(r"^<h[1-3]>", block):
            html_blocks.append(block)
        else:
            formatted_paragraph = block.replace("\n", " ")
            html_blocks.append(f"<p>{formatted_paragraph}</p>")

    return "\n\n".join(html_blocks)