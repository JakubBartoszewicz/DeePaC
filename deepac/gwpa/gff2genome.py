import re


def gff2genome(gff3_path, out_path):
    """Generate a .genome file."""
    ptrn = re.compile(r'(Genbank|RefSeq)\s+region')
    out_lines = []
    with open(gff3_path) as in_file:
        for line in in_file:
            region = ptrn.search(line)
            if region:
                out_lines.append(line.split()[0] + "\t" + line.split()[4] + "\n")
    with open(out_path, 'w') as out_file:
        out_file.writelines(out_lines)