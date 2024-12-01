def process(s):
    s = s.split(' ')[0]
    s = [x for x in s if x.isalnum()]
    return ''.join(s)

csv_file = "data.tsv"
run_file = 'run.sh'

# with open(run_file, 'w+') as f:
#     f.write()

with open(csv_file, 'r') as f:
    rows = f.readlines()

# focus_paper,opp_paper,topic,title_focus,title_opp,notes
for row in rows[1:]:
    cols = row.split('\t')
    focus_paper = cols[0]
    cited_paper = cols[1]
    topic = cols[2]
    shorthand = process(cols[3]) + "_" + process(cols[4])

    with open(run_file, 'a+') as f:
        f.write(f'focus_paper=\"{focus_paper}\"\n')
        f.write(f'cited_paper=\"{cited_paper}\"\n')
        f.write(f'log_dir=\"logs/{shorthand}/\"\n')
        f.write(f'CUDA_VISIBLE_DEVICES=2,3 python tree_of_debate.py --focus_paper $focus_paper --cited_paper $cited_paper --log_dir $log_dir --topic \"{topic}\"\n\n')

with open(run_file, 'a+') as f:
    f.write('notify \"tod\"\n')