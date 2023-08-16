
def count_width(s, align_zh):
    s = str(s)

    count = 0
    for ch in s:
        if align_zh and u'\u4e00' <= ch <= u'\u9fff':  # 中文占两格
            count += 2
        else:
            count += 1

    return count

def print_dict_to_md_table(dict):
    columns, rows = [], []
    for key, value in dict.times():
            columns +=  [key]
            rows += [value]
    print_to_markdwon_table(columns, [rows])
    
def print_to_markdwon_table(column, rows, align_zh = False):

    widths = []
    column_str = ""
    separate = "----"
    separate_str = ""
    for ci, cname  in enumerate(column):
        cw = count_width(cname, align_zh)
        for row in rows:
            item = row[ci]

            if count_width(item, align_zh) > cw:
                cw = count_width(item, align_zh)

        widths.append(cw)

        delete_count = count_width(cname, align_zh) - count_width(cname, False)

        column_str += f'|{cname:^{cw-delete_count+2}}'
        separate_str += f'|{separate:^{cw+2}}'

    column_str += "|"
    separate_str += "|"

    print(column_str)
    print(separate_str)

    for ri, row in enumerate(rows):
        row_str = ""
        for ci, item in enumerate(row):
            cw = widths[ci]

            delete_count = count_width(item, align_zh) - count_width(item, False)
            row_str += f'|{item:^{cw-delete_count+2}}'

        row_str += "|"
        print(row_str)
