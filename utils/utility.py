import os
import csv


def create_csv_writer(csv_file, column_names, sep=",", write_header=True):
    csv_excel_dialect = csv.Dialect
    csv_excel_dialect.delimiter = sep
    csv_excel_dialect.lineterminator = '\n'
    csv_excel_dialect.quoting = csv.QUOTE_NONE
    writer = csv.DictWriter(csv_file, fieldnames=column_names, dialect=csv_excel_dialect)
    if write_header:
        writer.writeheader()
    return writer


def create_kitti_csv_writer(out_dir):
    fname = "labels.csv"
    out_fp = os.path.join(out_dir, fname)
    columns = ["file_name", "class"]

    file = open(out_fp, "w")
    file_writer = create_csv_writer(file, columns, sep=",", write_header=True)
    return file_writer
