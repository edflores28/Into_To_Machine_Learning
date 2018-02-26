import preprocess
import feature_selection
import argparse

'''
This method executes the iris SFS kmeans.
'''
def do_iris1():
    print("The iris dataset will be processed and tested.")
    # Read the data for the specified file
    dataset = preprocess.read_file("iris.data")
    # Swap the specified columns in order to have the classifications
    # in the first column.
    dataset = preprocess.swap_columns(0,4,dataset,'False')
    data = preprocess.transpose(dataset)
    data = preprocess.transpose(data[1:])
    for entry in range(len(data)):
        data[entry] = [float(s.replace(',','')) for s in data[entry]]
    # Perform SFS + HAC
    print("Performing SFS with kmeans")
    feature_selection.SFS(dataset=data,k=3,)

'''
This method executes the iris SFS HAC.
'''
def do_iris2():
    print("The iris dataset will be processed and tested.")
    # Read the data for the specified file
    dataset = preprocess.read_file("iris.data")
    # Swap the specified columns in order to have the classifications
    # in the first column.
    dataset = preprocess.swap_columns(0,4,dataset,'False')
    data = preprocess.transpose(dataset)
    data = preprocess.transpose(data[1:])
    for entry in range(len(data)):
        data[entry] = [float(s.replace(',','')) for s in data[entry]]
    # Perform SFS + HAC
    print("Performing SFS with HAC")
    feature_selection.SFS(dataset=data,k=3,kmeans=False)

'''
This method executes the iris GA kmeans.
'''
def do_iris3():
    print("The iris dataset will be processed and tested.")
    # Read the data for the specified file
    dataset = preprocess.read_file("iris.data")
    # Swap the specified columns in order to have the classifications
    # in the first column.
    dataset = preprocess.swap_columns(0,4,dataset,'False')
    data = preprocess.transpose(dataset)
    data = preprocess.transpose(data[1:])
    for entry in range(len(data)):
        data[entry] = [float(s.replace(',','')) for s in data[entry]]
    # Perform GA + k means
    print("Performing GA with k menas")
    feature_selection.genetic_algorithm(data,8,5,3)

'''
This method executes the iris GA HAC.
'''
def do_iris4():
    print("The iris dataset will be processed and tested.")
    # Read the data for the specified file
    dataset = preprocess.read_file("iris.data")
    # Swap the specified columns in order to have the classifications
    # in the first column.
    dataset = preprocess.swap_columns(0,4,dataset,'False')
    data = preprocess.transpose(dataset)
    data = preprocess.transpose(data[1:])
    for entry in range(len(data)):
        data[entry] = [float(s.replace(',','')) for s in data[entry]]
    # Perform GA + k means
    print("Performing GA with HAC")
    feature_selection.genetic_algorithm(data,8,5,3,False)

'''
This method executes the glass SFS kmeans.
'''
def do_glass1():
    print("The glass dataset will be processed and tested.")
    # Read the data for the specified file
    dataset = preprocess.read_file("glass.data")
    dataset = preprocess.swap_columns(0,10,dataset)
    data = preprocess.transpose(dataset)
    data = preprocess.transpose(data[1:])
    for entry in range(len(data)):
        data[entry] = [float(s.replace(',','')) for s in data[entry]]
    # Perform SFS + HAC
    print("Performing SFS with kmeans")
    feature_selection.SFS(dataset=data,k=6)

'''
This method executes the glass SFS HAC.
'''
def do_glass2():
    print("The glass dataset will be processed and tested.")
    # Read the data for the specified file
    dataset = preprocess.read_file("glass.data")
    dataset = preprocess.swap_columns(0,10,dataset)
    data = preprocess.transpose(dataset)
    data = preprocess.transpose(data[1:])
    for entry in range(len(data)):
        data[entry] = [float(s.replace(',','')) for s in data[entry]]
    # Perform SFS + HAC
    print("Performing SFS with HAC")
    feature_selection.SFS(dataset=data,k=6,kmeans=False)

'''
This method executes the glass GA kmeans.
'''
def do_glass3():
    print("The glass dataset will be processed and tested.")
    # Read the data for the specified file
    dataset = preprocess.read_file("glass.data")
    dataset = preprocess.swap_columns(0,10,dataset)
    data = preprocess.transpose(dataset)
    data = preprocess.transpose(data[1:])
    for entry in range(len(data)):
        data[entry] = [float(s.replace(',','')) for s in data[entry]]
    # Perform GA + k means
    print("Performing GA with k menas")
    feature_selection.genetic_algorithm(data,8,5,6)

'''
This method executes the glass GA HAC.
'''
def do_glass4():
    print("The glass dataset will be processed and tested.")
    # Read the data for the specified file
    dataset = preprocess.read_file("glass.data")
    dataset = preprocess.swap_columns(0,10,dataset)
    data = preprocess.transpose(dataset)
    data = preprocess.transpose(data[1:])
    for entry in range(len(data)):
        data[entry] = [float(s.replace(',','')) for s in data[entry]]
    # Perform GA + k means
    print("Performing GA with HAC")
    feature_selection.genetic_algorithm(data,8,5,6,False)

'''
This method executes the spambase SFS kmeans.
'''
def do_spam1():
    print("The spambase dataset will be processed and tested.")
    # Read the data for the specified file
    dataset = preprocess.read_file("spambase.data")
    data = preprocess.transpose(dataset)
    data = preprocess.transpose(data[3:6])
    for entry in range(len(data)):
        data[entry] = [float(s.replace(',','')) for s in data[entry]]

    print("Performing SFS with kmeans")
    feature_selection.SFS(dataset=data,k=2)

'''
This method executes the spambase SFS HAC.
'''
def do_spam2():
    print("The spambase dataset will be processed and tested.")
    # Read the data for the specified file
    dataset = preprocess.read_file("spambase.data")
    data = preprocess.transpose(dataset)
    data = preprocess.transpose(data[4:5])
    for entry in range(len(data)):
        data[entry] = [float(s.replace(',','')) for s in data[entry]]
    # Perform SFS + HAC
    print("Performing SFS with HAC")
    feature_selection.SFS(dataset=data,k=2,kmeans=False)

'''
This method executes the spambase GA kmeans.
'''
def do_spam3():
    print("The spambase dataset will be processed and tested.")
    # Read the data for the specified file
    dataset = preprocess.read_file("spambase.data")
    data = preprocess.transpose(dataset)
    data = preprocess.transpose(data[:6])
    for entry in range(len(data)):
        data[entry] = [float(s.replace(',','')) for s in data[entry]]
    # Perform GA + k means
    print("Performing GA with k menas")
    feature_selection.genetic_algorithm(data,8,5,2)

'''
This method executes the spambase GA HAC.
'''
def do_spam4():
    print("The spambase dataset will be processed and tested.")
    # Read the data for the specified file
    dataset = preprocess.read_file("spambase.data")
    data = preprocess.transpose(dataset)
    data = preprocess.transpose(data[3:4])
    for entry in range(len(data)):
        data[entry] = [float(s.replace(',','')) for s in data[entry]]
    # Perform GA +  means
    print("Performing GA with HAC")
    feature_selection.genetic_algorithm(data,8,5,2,False)
'''
MAIN
'''
# Create a parser for the command line arguments
parser = argparse.ArgumentParser(description="Intro to ML Project 2")
parser.add_argument('-i1',action="store_true", default=False, help='Execute iris test SFS+kmeans')
parser.add_argument('-i2',action="store_true", default=False, help='Execute iris test SFS+HAC')
parser.add_argument('-i3',action="store_true", default=False, help='Execute iris test GA+kmeans')
parser.add_argument('-i4',action="store_true", default=False, help='Execute iris test GA+HAC')
parser.add_argument('-g1',action="store_true", default=False, help='Execute glass test SFS+kmeans')
parser.add_argument('-g2',action="store_true", default=False, help='Execute iris test SFS+HAC')
parser.add_argument('-g3',action="store_true", default=False, help='Execute glass test GA+kmeans')
parser.add_argument('-g4',action="store_true", default=False, help='Execute glass test GA+HAC')
parser.add_argument('-s1',action="store_true", default=False, help='Execute spambase test SFS+kmeans')
parser.add_argument('-s2',action="store_true", default=False, help='Execute spambase test SFS+HAC')
parser.add_argument('-s3',action="store_true", default=False, help='Execute spambase test GA+kmeans')
parser.add_argument('-s4',action="store_true", default=False, help='Execute spambase test GA+HAC')

results = parser.parse_args()

# Perform the tests based on the input
if results.i1:
    do_iris1()
if results.i2:
    do_iris2()
if results.i3:
    do_iris3()
if results.i4:
    do_iris4()
if results.g1:
    do_glass1()
if results.g2:
    do_glass2()
if results.g3:
    do_glass3()
if results.g4:
    do_glass4()
if results.s1:
    do_spam1()
if results.s2:
    do_spam2()
if results.s3:
    do_spam3()
if results.s4:
    do_spam4()
