import preprocess


def do_ecoli():
    dataset = preprocess.read_file("ecoli.data")
    preprocess.create_partitions(5,8,dataset)
do_ecoli()
