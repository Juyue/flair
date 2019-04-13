import sys

def parse_csv_line(l) :
    l = l.strip().split(",")
    fn,l1,l2,typ,sco = l[0],l[1],l[2],l[3],l[4]
    sur = ",".join(l[5:]).strip("\"")
    return fn,l1,l2,typ,sco,sur

def merger_lines(fname) :
    with open(fname, encoding='utf-8') as f :
        res = f.readline().strip()
        fn,l1,l2,typ,sco,sur=parse_csv_line(res)
        for line in f :
            next_res = line.strip()
            _fn,_l1,_l2,_typ,_sco,_sur=parse_csv_line(next_res)
            if fn==_fn and int(l2)+1==int(_l1) and typ[-3:]==_typ[-3:] and _typ[:2]!='B-' :
                l2=_l1
                sur = " ".join([sur,_sur])
            else :
                sur = "".join(["\"",sur,"\""])
                print(",".join([fn,l1,l2,typ[-3:],sco,sur]))
                fn,l1,l2,typ,sco,sur=parse_csv_line(next_res)
        sur = "".join(["\"",sur,"\""])
        print(",".join([fn,l1,l2,typ[-3:],sco,sur]))

if __name__ == "__main__" :
    fname = sys.argv[1]
    merger_lines(fname)


