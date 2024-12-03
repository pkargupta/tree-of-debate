prefix = "CUDA_VISIBLE_DEVICES=0,1"

def process(s):
    # s = ''.join(s.split(' ')[:2])
    # s = [x for x in s if x.isalnum()]
    # return ''.join(s).lower()
    return s[s.rfind('/')+1:].replace('.', '_')

csv_file = "data.tsv"
run_file = 'run.sh'

# with open(run_file, 'w+') as f:
#     f.write()

with open(csv_file, 'r') as f:
    rows = f.readlines()

# focus_paper,opp_paper,topic,title_focus,title_opp,notes
def tod_runs(ret):
    if ret:
        tod_file = "tree_of_debate"
    else:
        tod_file = "tree_of_debate_no_ret"

    for row in rows[1:]:
        cols = row.split('\t')
        focus_paper = process(cols[0])
        cited_paper = process(cols[1])
        topic = cols[2]
        # shorthand = process(cols[0]) + "-" + process(cols[1])

        with open(run_file, 'a+') as f:
            f.write(f'focus_paper=\"{focus_paper}\"\n')
            f.write(f'cited_paper=\"{cited_paper}\"\n')
            f.write(f'{prefix} python {tod_file}.py --focus_paper $focus_paper --cited_paper $cited_paper --topic \"{topic}\"\n\n')


####################
with open(run_file, 'a+') as f:
    f.write("########### TREE-OF-DEBATE RUNS ###########\n")
tod_runs(True)
with open(run_file, 'a+') as f:
    f.write("########### TREE-OF-DEBATE RUNS ###########\n\n\n\n")


####################
with open(run_file, 'a+') as f:
    f.write("########### BASELINES RUNS ###########\n")
    f.write('cd baselines\npython data_processor.py\nsource run.sh\ncd ..\n')
    f.write("########### BASELINES RUNS ###########\n\n\n\n")
####################
    
with open(run_file, 'a+') as f:
    f.write("########### TREE-OF-DEBATE ABLATION: NO RETRIEVAL RUNS ###########\n")
tod_runs(False)
with open(run_file, 'a+') as f:
    f.write("########### TREE-OF-DEBATE ABLATION: NO RETRIEVAL RUNS ###########\n\n\n\n")

####################
with open(run_file, 'a+') as f:
    f.write('notify \"tod\"\n')

####################
    
with open(run_file, 'a+') as f:
    f.write("########### EVALUATION ###########\n")
for row in rows[1:]:
    cols = row.split('\t')
    focus_paper = process(cols[0])
    cited_paper = process(cols[1])
    topic = cols[2]
    # shorthand = process(cols[0]) + "-" + process(cols[1])

    with open(run_file, 'a+') as f:
        f.write(f'# focus_paper=\"{focus_paper}\"\n')
        f.write(f'# cited_paper=\"{cited_paper}\"\n')
        f.write(f'# python side_by_side.py --focus_paper $focus_paper --cited_paper $cited_paper --topic \"{topic}\"\n\n')
with open(run_file, 'a+') as f:
    f.write("########### EVALUATION ###########\n\n\n\n")