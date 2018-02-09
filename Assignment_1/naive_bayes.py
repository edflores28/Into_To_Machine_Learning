P = 0.001

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
    pct_list = [[] for i in range(2)]
    pct_list[0].append(0)
    pct_list[1].append(0)
    pct_list[0] += [[0,0] for i in range(len(table_list[0][1:]))]
    pct_list[1] += [[0,0] for i in range(len(table_list[0][1:]))]
    for row in table_list:
        print(row)
        pct_list[row[0]][0] += 1
        for i in range(1, len(row)):
            pct_list[row[0]][i][row[i]]+=1
    conv_pct(pct_list[0])
    conv_pct(pct_list[1])
    for i in range(len(pct_list)):
        pct_list[i][0] /= len(table_list)

    return pct_list

'''
This method takes a row in the data set along with the
calculated percent table and determines which value
is mostly likley to occur
'''
def calculate(row, length, pct):
    one_calc = pct[0][0]
    zero_calc = pct[1][0]
    for i in range(1,len(row)):
        one_calc *= pct[1][i][row[i]]
        zero_calc *= pct[0][i][row[i]]
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
    for i in range(len(table_list)):
        pred = calculate(table_list[i], len(table_list), pct_table)
        if pred == table_list[i][0]:
            results[1] += 1
        else:
            results[0] += 1
    print("Naive Bayes Statistics:")
    print("Total successfully matched:\t", results[1])
    print("Total successfully not matched:\t", results[0])
    print("Percentage matched:\t", (results[1]*100)/(results[1]+results[0]),"\n")
