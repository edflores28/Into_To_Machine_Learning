import math
import random
import cluster
import silhoutte

MUTATE_PROB = 0.07

'''
Transpose the datalist from rows to columns or
columns to rows
'''
def transpose(data_list):
    return list(map(list,zip(*data_list)))
'''
The stepwise forward selection algorithm. This algorithms
attemps to find a set of best features of the data Set
'''
def SFS(dataset,k,kmeans=True):
    total_features = len(dataset)
    best_features = []
    baseperf = -math.inf
    currperf = -math.inf
    best_index = 0
    data_set = transpose(dataset)
    # Break the while loop when there is no more features
    # in the dataset
    while len(data_set) != 0:
        bestperf = -math.inf
        # Iterate through all available features append
        # obtain a cluster list (learning) and calculate
        # the error with the silhoutte coefficent.
        # Update statistics
        for feature in range(len(data_set)):
            best_features.append(data_set[feature])
            # Obtain a row representation of the features
            # The translates to coordinate positions of there
            # data.
            best_features = transpose(best_features)
            if kmeans == True:
                clusters = cluster.k_means(best_features,k)
            else:
                clusters = cluster.HAC(best_features,k)
            currperf = silhoutte.performance(best_features, clusters)

            print ("Performance of feature", feature, "is", currperf)
            # Determine if the current performance is greater
            # than the best performace. If so store the columns
            # index and update the best performace
            if currperf > bestperf:
                best_index = feature
                bestperf = currperf
                print("Feature", best_index,"chosen.")
            # Obtain the column representation of the data
            # and remove the feature.
            best_features = transpose(best_features)
            best_features.pop(len(best_features)-1)
        # When the best performace is greater than the base
        # performace, update the base performance, store there
        # feature set, and remove the feature set from there
        # main feature set
        if bestperf > baseperf:
            print("Base performance: ", baseperf, "Best performance: ", bestperf)
            print("Feature",best_index,"added to best features")
            baseperf = bestperf
            best_features.append(data_set[best_index])
            data_set.pop(best_index)
        # Exit when there is no improvement
        else:
            print("Finished with ", baseperf)
            break
    print("Forward Selection Complete")
    print("Calculated Performance: ", baseperf)

'''
The fitness evaluation for the genetic algorithm. This
method performs clustering on each individual in then
given population and calulates the silhoutte coefficent
to determine the fitness of the individual
'''
def evaluate_fitness(population, X, k, kmeans=True):
    fitness = [0.0 for i in range(len(population))]
    # Transpose X for a column representation
    X_col = transpose(X)
    # Traverse through each population entry
    # and build a temporary list of features
    # to evalute
    for row in range(len(population)):
        temp = []
        for column in range(len(population[row])):
            if population[row][column] == 1:
                temp.append(X_col[column])
        # Transpose temp for a row representation
        temp = transpose(temp)
        # Only cluster and evalute if there is at
        # least one feature in the individual
        if population[row].count(1) >= 1:
            if kmeans == True:
                clusters = cluster.k_means(temp,k)
            else:
                clusters = cluster.HAC(temp,k)
            fitness[row] = silhoutte.performance(temp, clusters)
        else:
            fitness[row] = 0.0
    return fitness

'''
This method takes two parents and creates two children
using just one crossover point.
'''
def crossover(parent_a, parent_b):
    cross_point = random.randint(1,len(parent_a))
    print("Creating offspring, cross point is:", cross_point)
    child_a = parent_a[:cross_point] + parent_b[cross_point:]
    child_b = parent_b[:cross_point] + parent_a[cross_point:]
    return child_a, child_b

'''
This method iterates through each gene of the individual
and mutates the gene if a random number is below MUTATE_PROB.
'''
def mutate(child):
    print("Mutating child.")
    for entry in range(len(child)):
        if random.uniform(0,1) <= MUTATE_PROB:
            if child[entry] == 1:
                child[entry] = 0
            else:
                child[entry] = 1
    return child

'''
This method takes the newly created children and replaces
them with the least fittest individuals in the population.
Once replced the fitness values are also updated.
'''
def replace(pop, pop_fitness, children, child_fitness):
    temp = pop_fitness
    # Obtain the least fittest individual
    least_fit = temp.index(min(temp))
    # Set least fit to a high number to obtain the
    # second least fittest
    temp[least_fit] = 1.0
    second_fit = temp.index(min(temp))
    # Replace the least fittest in the population
    # with the new children and update the fitness
    pop[least_fit] = children[0]
    pop[second_fit] = children[1]
    pop_fitness[least_fit] = child_fitness[0]
    pop_fitness[second_fit] = child_fitness[1]
    print ("Replacing least fit individuals:")
    print(pop[least_fit])
    print(pop[second_fit])
    print("With the following children:")
    for entry in children:
        print (entry)
    return pop, pop_fitness

'''
The genetic algorithm geared for feature selection.
This method creates a random population and evaluates
the fitness of the populations. Then the process of
selection, crossover, mutation, and replacement parents
dont until a set number of generations.
'''
def genetic_algorithm(X, pop_size, generations, k, kmeans=True):
    pop = random.sample(range(1, int(math.pow(2,len(X[0])))-1), pop_size)
    gen_count = 0
    # Convert the population to binary representation
    for entry in range(len(pop)):
        temp = bin(pop[entry])[2:]
        pop[entry] = [0 for i in range(len(X[0]) - len(temp))]
        pop[entry] += list(map(int,temp))
    print("Evaluating initial fitness")
    fitness = evaluate_fitness(pop,X,k,kmeans)
    while gen_count < generations:
        # Select 2 parents at random
        parents = random.sample(pop,2)
        # Perform crossover on the parents to create children
        child_a, child_b = crossover(parents[0],parents[1])
        # Mutate both children
        children = [mutate(child_a), mutate(child_b)]
        # Evaluate the fitness of the children
        child_fitness = evaluate_fitness(children, X, k,kmeans)
        # Replace the two least fittest individuals in the population
        pop, fitness = replace(pop, fitness, children, child_fitness)
        gen_count += 1
        print("Created generation",gen_count)
    # Print the fittest individual
    value = max(fitness)
    print("The most fittest individual has fitness:", value)
    print(pop[fitness.index(value)])
