"""Export assignment3/report.md to HTML and PDF the same way as the A1 report:
python-markdown with the A1 stylesheet, then headless Chrome."""
import subprocess

import markdown

from data import A3

STYLE = """<meta charset='utf-8'><style>
body{font-family:Helvetica,Arial,sans-serif;font-size:11pt;line-height:1.45;max-width:17cm;margin:2cm auto;color:#222}
h1,h2{border-bottom:1px solid #ccc;padding-bottom:.2em} h2{page-break-before:auto} h1,h2,h3{break-after:avoid}
table{border-collapse:collapse;margin:1em 0;font-size:9.5pt} th,td{border:1px solid #ccc;padding:4px 8px;text-align:left}
th{background:#f2f2f2} code{background:#f4f4f4;padding:1px 4px;font-size:9.5pt} pre{background:#f4f4f4;padding:8px;font-size:9pt}
img{max-width:100%} em{color:#555} hr{border:0;border-top:1px solid #ccc}
</style>"""
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"


def main():
    source = (A3 / "report.md").read_text()
    html = STYLE + markdown.markdown(source, extensions=["tables", "fenced_code"])
    (A3 / "report.html").write_text(html)
    subprocess.run([CHROME, "--headless", "--disable-gpu", "--no-pdf-header-footer",
                    f"--print-to-pdf={A3 / 'report.pdf'}", (A3 / "report.html").as_uri()],
                   check=True, capture_output=True)
    print("wrote", A3 / "report.html", A3 / "report.pdf")


if __name__ == "__main__":
    main()
