# Constant variables
MAX_ROWS_PRINT = 15
PRINT_COUNTER = 0;

def conv_pct(list):
    for i in range(1,len(list)):
        list[i][0]= list[i][0]/list[0]
        list[i][1]=list[i][1]/list[0]

'''
This takes the data table and performs the naive bayes
calculation and returns this calculation in a table
format
'''
def build_table(table_list):
    print("Building Naive Bayes percentage table.\n")
    pct_list = [[] for index in range(2)]
    pct_list[0].append(0)
    pct_list[1].append(0)
    pct_list[0] += [[0,0] for index in range(len(table_list[0][1:]))]
    pct_list[1] += [[0,0] for index in range(len(table_list[0][1:]))]
    for row in table_list:
        pct_list[row[0]][0] += 1
        for entry in range(1, len(row)):
            pct_list[row[0]][entry][row[entry]]+=1
    conv_pct(pct_list[0])
    conv_pct(pct_list[1])
    for index in range(len(pct_list)):
        pct_list[index][0] /= len(table_list)

    print("The following are the percent tabe built note values inside the bracket [X,Y].")
    print("X = the percentage of 0 values and Y = percentage of 1 values.\n")
    print ("Zero Row: ")
    print(pct_list[0],"\n")
    print ("One Row: ")
    print(pct_list[1],"\n")
    return pct_list

'''
This method takes a row in the data set along with the
calculated percent table and determines which value
is mostly likley to occur
'''
def calculate(row, length, pct):
    one_calc = pct[0][0]
    zero_calc = pct[1][0]
    for index in range(1,len(row)):
        one_calc *= pct[1][index][row[index]]
        zero_calc *= pct[0][index][row[index]]
    if one_calc > zero_calc:
        return 1
    else:
        return 0

'''
This method takes the data and pct tables and
makes a prediction and reports the Statistics
'''
def test(table_list, pct_table):
    results = [0,0]
    for index in range(len(table_list)):
        pred = calculate(table_list[index], len(table_list), pct_table)
        if pred == table_list[index][0]:
            results[1] += 1
        else:
            results[0] += 1
    print("Naive Bayes Statistics:")
    print("Total matched:\t\t", results[1])
    print("Total not matched:\t", results[0])
    print("Percentage matched:\t", (results[1]*100)/(results[1]+results[0]),"\n")
