import random
import math
import time
import networkx as nx      
import matplotlib.pyplot as plt
#All the functions implemented below are written by Tevfik Buğra Türker.
#Libraries are used only for randomness, time keeping and visualizing purposes.

#------  Other Functions   ------------------------------------------------------------------------------------
def sort_dict_by_value_reversed(my_dict):
    # This function creates a list of keys sorted by their corresponding values in increasing order
    dict_items = my_dict.items()
    sorted_items = sorted(dict_items, key=lambda x: x[1], reverse=True)
    sorted_keys = [item[0] for item in sorted_items]
    return sorted_keys

def sort_dict_by_value(my_dict):
    # This function creates a list of keys sorted by their corresponding values in decreasing order
    dict_items = my_dict.items()
    sorted_items = sorted(dict_items, key=lambda x: x[1])
    sorted_keys = [item[0] for item in sorted_items]
    return sorted_keys

def check_graphicality(seq_dict):
    #This function checks the graphicality of a given degree sequence dictionary.
    #This function is required for sequential algorithm to work.
    deg_seq = list()
    for key in list(seq_dict.keys()):
        deg_seq.append(seq_dict[key])
    if sum(deg_seq) %2 == 1:                                #check if the sum of the sequence is even
        return False
    sorted_degree_sequence = sorted(deg_seq, reverse=True)  #sort the sequence
    for k in range(1,len(deg_seq)+1):
        l = list()
        for a in sorted_degree_sequence[k:len(deg_seq)+1]:
            l.append(min(a,k))
        if sum(sorted_degree_sequence[0:k]) > k*(k-1) + sum(l):     #Erdös-Gallai
            return False
    return True

def visualize(adj_dict):
    #This function plots the specified adjacency dictionary
    G = nx.Graph()                                              # Create a graph object
    G.add_nodes_from([i for i in range(1,len(adj_dict)+1)])     # Add nodes to the graph
    edges = adj_dict
    for node in edges:                                          # Add edges to the graph
        for neighbor in edges[node]:
            G.add_edge(node, neighbor)
  
    nx.draw(G, with_labels=True)                                # Draw the graph
    plt.show()

def breadth_first_search(adj_d):
    #this funtion is used for deciding the connectedness of a graph and returns True if the graph is connected and False otherwise
    #it takes the input as an adjacency dictionary
    Q = [1]                     #start checking from vertex 1 - vertices to be checked
    labeled_vertices = []       #add labeled vetices here
    M = []                      #vertices reachable from vertex 1
    while Q != []:
        v = Q[0]
        del Q[0]
        labeled_vertices.append(v)
        M.append(v)
        for i in adj_d[v]:
            if i not in labeled_vertices and i not in Q:
                Q.append(i)
    return sorted(M) == list(adj_d.keys())


#------  Graphical Random Generation Functions   ----------------------------------------------------------------
def graphical_random_generator(n):
    #this function generates a graphical random degree sequence of length n
    degree_sequence = [None] * n
    for i in range(n):
        degree_sequence[i] = random.randint(1, n-1)                 #we omit 0 here, since it creates disconnected vertices and pairwise exchange cannot make it connected
    if sum(degree_sequence) %2 == 1:                                #check if the sum of the sequence is even
        return graphical_random_generator(n)
    sorted_degree_sequence = sorted(degree_sequence, reverse=True)  #sort the sequence
    for k in range(1,n+1):
        l = list()
        for a in sorted_degree_sequence[k:n+1]:
            l.append(min(a,k))
        if sum(sorted_degree_sequence[0:k]) > k*(k-1) + sum(l):     #Erdös-Gallai
            return graphical_random_generator(n)
        
    return sorted_degree_sequence

#m=(graphical_random_generator(6))
#print(m)

def exponential_random_number(lambd):
    u = random.random()                         # Generate a random number between 0 and 1
    x = -math.log(1 - u) / lambd                # Compute the inverse transform
    return math.ceil(x)                         #Get the ceiling of the random exponential value

def graphical_exponential_generator(n):
    #this function generates a graphical random exponential(lambda = 2/n) degree sequence of length n
    degree_sequence = [None] * n
    for i in range(n):
        degree_sequence[i] = exponential_random_number(2/n)        #lambda = 2/n    --> expected value is n/2
        while degree_sequence[i] > n-1:
            degree_sequence[i] = exponential_random_number(2/n)
    if sum(degree_sequence) %2 == 1:                                #check if the sum of the sequence is even
        return graphical_exponential_generator(n)
    sorted_degree_sequence = sorted(degree_sequence, reverse=True)  #sort the sequence
    for k in range(1,n+1):
        l = list()
        for a in sorted_degree_sequence[k:n+1]:
            l.append(min(a,k))
        if sum(sorted_degree_sequence[0:k]) > k*(k-1) + sum(l):     #Erdös-Gallai
            return graphical_exponential_generator(n)
        
    return sorted_degree_sequence

#m=(graphical_exponential_generator(50))
#print(m)

#------  Graph Construction Algorithms   -----------------------------------------------------------------------
def graph_generation_HH_random_to_high(seq):
    #Havel-Hakimi Random Vertex Selection, distributed to the Highest Degree
    
    seq_dict = dict()                                               #sequence dictionary
    for v in range(1,len(seq)+1):
        seq_dict[v] = seq[v-1]
    adjacency_dict = dict()                                         #adjacency dictionary
    for v in range(1,len(seq)+1):
        adjacency_dict[v] = list()
    
    for q in range(len(seq_dict)):
        key_vertex = random.choice(list(seq_dict.keys()))           #select vertices randomly
        degree_vertex = seq_dict[key_vertex]
        del seq_dict[key_vertex]                                    #delete vertex to be used from the dictionary
        keys_of_values_inc = sort_dict_by_value_reversed(seq_dict)  #sort vertices in increasing degrees
        for o in keys_of_values_inc[0:degree_vertex]:
            seq_dict[o] -= 1                                        #subtract 1 from appropriate vertices
            adjacency_dict[key_vertex].append(o)                    #construct adjacency dictionary
            adjacency_dict[o].append(key_vertex)
    
    for n in list(adjacency_dict.keys()):                           
        adjacency_dict[n] = sorted(adjacency_dict[n])               #sort adjacency list
        if len(adjacency_dict[n]) != seq[n-1]:
            print("This algorithm cannot build a graph with the given sequence.")
            return False                                            #check if intended degree sequences are obtained, if not quit
    return adjacency_dict

#print(graph_generation_HH_random_to_high(m))

def graph_generation_HH_highest_to_high(seq):
    #Havel-Hakimi Highest Degree Vertex First, distributed to the Highest Degree
    seq_dict = dict()                                               #sequence dictionary
    for v in range(1,len(seq)+1):
        seq_dict[v] = seq[v-1]
    adjacency_dict = dict()                                         #adjacency dictionary
    for v in range(1,len(seq)+1):
        adjacency_dict[v] = list()

    for q in range(len(seq_dict)):
        key_vertex = sort_dict_by_value_reversed(seq_dict)[0]       #select vertices of highest degree first
        degree_vertex = seq_dict[key_vertex]
        del seq_dict[key_vertex]                                    #delete vertex to be used from the dictionary
        keys_of_values_inc = sort_dict_by_value_reversed(seq_dict)  #sort vertices in increasing degrees
        for o in keys_of_values_inc[0:degree_vertex]:
            seq_dict[o] -= 1                                        #subtract 1 from appropriate vertices
            adjacency_dict[key_vertex].append(o)                    #construct adjacency dictionary
            adjacency_dict[o].append(key_vertex)
    
    for n in list(adjacency_dict.keys()):                           
        adjacency_dict[n] = sorted(adjacency_dict[n])               #sort adjacency list
        if len(adjacency_dict[n]) != seq[n-1]:
            print("This algorithm cannot build a graph with the given sequence.")
            return False                                            #check if intended degree sequences are obtained, if not quit
    return adjacency_dict    

#a = (graph_generation_HH_highest_to_high(m))
#print(a)

def graph_generation_HH_smallest_to_high(seq):
    #Havel-Hakimi Smallest Degree Vertex First, distributed to the Highest Degree
    seq_dict = dict()                                               #sequence dictionary
    for v in range(1,len(seq)+1):
        seq_dict[v] = seq[v-1]
    adjacency_dict = dict()                                         #adjacency dictionary
    for v in range(1,len(seq)+1):
        adjacency_dict[v] = list()

    for q in range(len(seq_dict)):
        key_vertex = sort_dict_by_value(seq_dict)[0]                #select vertices of lowest degree first
        degree_vertex = seq_dict[key_vertex]
        del seq_dict[key_vertex]                                    #delete vertex to be used from the dictionary
        keys_of_values_inc = sort_dict_by_value_reversed(seq_dict)  #sort vertices in increasing degrees
        for o in keys_of_values_inc[0:degree_vertex]:
            seq_dict[o] -= 1                                        #subtract 1 from appropriate vertices
            adjacency_dict[key_vertex].append(o)                    #construct adjacency dictionary
            adjacency_dict[o].append(key_vertex)
    
    for n in list(adjacency_dict.keys()):                           
        adjacency_dict[n] = sorted(adjacency_dict[n])               #sort adjacency list
        if len(adjacency_dict[n]) != seq[n-1]:
            print("This algorithm cannot build a graph with the given sequence.")
            return False                                            #check if intended degree sequences are obtained, if not quit
    return adjacency_dict  

#print(graph_generation_HH_smallest_to_high(m))

def graph_generation_HH_smallest_to_small(seq):
    #Havel-Hakimi Smallest Degree Vertex First, distributed to the Smallest Degree
    seq_dict = dict()                                               #sequence dictionary
    for v in range(1,len(seq)+1):
        seq_dict[v] = seq[v-1]
    adjacency_dict = dict()                                         #adjacency dictionary
    for v in range(1,len(seq)+1):
        adjacency_dict[v] = list()

    for q in range(len(seq_dict)):
        key_vertex = sort_dict_by_value(seq_dict)[0]                #select vertices of lowest degree first
        degree_vertex = seq_dict[key_vertex]
        del seq_dict[key_vertex]                                    #delete vertex to be used from the dictionary
        keys_of_values_inc = sort_dict_by_value(seq_dict)           #sort vertices in decreasing degrees
        for o in keys_of_values_inc[0:degree_vertex]:
            seq_dict[o] -= 1                                        #subtract 1 from appropriate vertices
            adjacency_dict[key_vertex].append(o)                    #construct adjacency dictionary
            adjacency_dict[o].append(key_vertex)
    
    for n in list(adjacency_dict.keys()):                           
        adjacency_dict[n] = sorted(adjacency_dict[n])               #sort adjacency list
        if len(adjacency_dict[n]) != seq[n-1]:
            #print("This algorithm cannot build a graph with the given sequence.")
            return False                                            #check if intended degree sequences are obtained, if not quit
    return adjacency_dict  
#a = graphical_random_generator(100)
#print(graph_generation_HH_smallest_to_small(a))

def sequential_algorithm(seq):
    #This function is the implementation of sequential algorithm.
    #As the input, specify the graphical degree sequence.
    E = []         #Empty list of edges
    seq_dict = dict()                                               #sequence dictionary
    for v in range(1,len(seq)+1):
        seq_dict[v] = seq[v-1]
    zero_dict = dict()                                              #dictionary of zeros to exit the loop
    for v in range(1,len(seq)+1):
        zero_dict[v] = 0
    adjacency_dict = dict()                                         #adjacency dictionary
    for v in range(1,len(seq)+1):
        adjacency_dict[v] = list()


    while seq_dict != zero_dict:                                    #iterate until there is no "residual degree" left
        min_pos_entry = float("inf")                                #track the minimum positive entry
        min_pos_key = float("inf")                                  #track the minimum key of the min positive entry
        for key in list(seq_dict.keys()):
            if seq_dict[key] < min_pos_entry and seq_dict[key] > 0: #find min_pos_entry
                min_pos_entry = seq_dict[key]
                min_pos_key = key
        candidates = list()                                         #define candidates list
        for key in list(seq_dict.keys()):                           #iterate through all the keys to find the candidates
            if key == min_pos_key:                                  #if i=j(as it is defined in the paper) continue
                continue
            if [key,min_pos_key] in E or [min_pos_key,key] in E or seq_dict[key] == 0:  #if this edge is already in E or key position is empty, then continue
                continue
            new_dict = seq_dict.copy()                              #create a copy of sequence dictionary    
            new_dict[key] -=1                                       
            new_dict[min_pos_key] -=1
            if check_graphicality(new_dict):                        #check if probable changes created a graphical sequence
                candidates.append(key)                              #if yes, the add it to the candidates list
        j = random.choices(candidates, weights=[seq_dict[j] for j in candidates])[0]    #choose j from the candidate list with probability proportional to its degree in d
        E.append((min_pos_key, j))                                  #append edge to E
        seq_dict[min_pos_key] -= 1                                  #update sequence dictionary
        seq_dict[j] -= 1
        adjacency_dict[min_pos_key].append(j)                       #construct adjacency dictionary
        adjacency_dict[j].append(min_pos_key)
    for n in list(adjacency_dict.keys()):                           #sort the adjacency dictionary for visual purposes
        adjacency_dict[n] = sorted(adjacency_dict[n])
    
    return adjacency_dict                                   #It is possible to return the Edge List(E), but for output uniformity, returning adjacency dict is preferred.

#a = sequential_algorithm(m)
#print(a)


#----- Function to Analyse Graph Genereation Ability of Algorithms-------------------------------------------------
def graph_generation_percentage(l,u,algorithm,k):
    # this function calculates the percantage of graphs genereted successfully when given a graphical degree sequence of lentgh n
    # l: lower bound for the length of the degree sequence,  u: upper bound for the length of the degree sequence,  algorithm: name of the algorithm,  k: number of trials
    generation_dict = dict()    
    for x in range(l,u+1):    
        cannot_construct = 0
        for i in range(k):
            deg_seq = graphical_random_generator(x)
            if algorithm(deg_seq) == False:
                cannot_construct +=1
            generation_dict[x] = 1 - cannot_construct/k
    x = list(generation_dict.keys())
    y = list(generation_dict.values())
    plt.plot(x, y, marker='o')
    plt.ylim(-0.05, 1.05)
    plt.xlabel('Number of Vertices')
    plt.ylabel('Percentage of Generating in 10000 Trials')
    plt.title('Graph Genereation Ability of HH Smallest to Small')
    plt.grid(True)
    plt.show()

#graph_generation_percentage(2,10,graph_generation_HH_smallest_to_small,10000)


#----- Function to Analyse Disconnected Graphs --------------------------------------------------------------------
def count_disconnected_graph(n,algorithm,k):
    #this function calculates how many disconnected graphs are generated in k trials.
    #inputs: n = length of the degree list, algorithm: specify the name of the algorithm, k = number of trials
    count=0
    disconnected_number = 0
    while count < k:
        degree_list = graphical_random_generator(n)
        degree_sequence = algorithm(degree_list)
        count+=1
        if breadth_first_search(degree_sequence) == False:
            disconnected_number +=1
    print("There are", disconnected_number,"disconnected graphs out of",k ,"trials found with",algorithm.__name__,".")    

#random.seed(2)
#count_disconnected_graph(50,graph_generation_HH_random_to_high,10000)
#count_disconnected_graph(100,graph_generation_HH_highest_to_high,10000)
#count_disconnected_graph(100,graph_generation_HH_smallest_to_high,10000)
#do not run this# count_disconnected_graph(10,graph_generation_HH_smallest_to_small,10000) #in most of the cases, this function cannot build a graph
#count_disconnected_graph(50,sequential_algorithm,1000)


#----- Pairwise Edge Interchange -----------------------------------------------------------------------------------
def pairwise_edge_interchange(adj_d):
    #This function makes pairwise edge interchanges and makes disconnected graphs connected.
    connected = breadth_first_search(adj_d)           #check if degree sequence yields a connected graph
    count = 0
    while not connected:                                #iterate until it's connected
        a, x = random.choice(list(adj_d.items()))     #choose a,b,c,d in a random manner
        b = random.choice(list(x))
        c, y = random.choice(list(adj_d.items()))
        d = random.choice(list(y))

        if a != c and b != d and c not in adj_d[a] and d not in adj_d[b]:       #check if chosen a,b,c,d satisfy some prerequisites
            adj_d[a].remove(b)                                                    
            adj_d[b].remove(a)
            adj_d[c].remove(d)
            adj_d[d].remove(c)
            adj_d[a].append(c)
            adj_d[c].append(a)
            adj_d[b].append(d)
            adj_d[d].append(b)
            
            count += 1
            connected = breadth_first_search(adj_d)
    for n in list(adj_d.keys()):                           #sort the adjacency dictionary for visual purposes
        adj_d[n] = sorted(adj_d[n])
    return  count, adj_d


#----- Measuring Time ---------------------------------------------------------------------------------------------
def measure_time(random_deg_seq, algorithm):
    # This function measures the time needed to 
    # 2.1.	Generate a graph from a degree sequence with the specified algorithm
    # 2.2.	Check for connectivity 
    # 2.3.	If the graph is not connected, do pairwise edge interchanges  

    start = time.time()
    adj_dict_generated = algorithm(random_deg_seq)
    connectivity = breadth_first_search(adj_dict_generated)
    if not connectivity:
        connected_adj_dict = pairwise_edge_interchange(adj_dict_generated)
    time_elapsed = time.time() - start

    return(time_elapsed)

#random.seed(0)
#random_deg_seq = graphical_random_generator(1000) #n=10

#print(measure_time(random_deg_seq,graph_generation_HH_random_to_high))
#print(measure_time(random_deg_seq,graph_generation_HH_highest_to_high))
#print(measure_time(random_deg_seq,graph_generation_HH_smallest_to_high))
###do not run it ---- print(measure_time(50,random_deg_seq,graph_generation_HH_smallest_to_small)) #most of the time this algorithm cannot build a graph
#print(measure_time(random_deg_seq,sequential_algorithm))

def measure_algorithm_k(n, algorithm,k):
    # This function measures the time needed for an algorithm to generate a graph from a degree sequence.
    # n = length of the degree sequence
    # k = number of trials 
    #choose the algorithm from these options:
    #   graph_generation_HH_random_to_high
    #   graph_generation_HH_highest_to_high
    #   graph_generation_HH_smallest_to_high
    #   graph_generation_HH_smallest_to_small #most of the time this algorithm cannot build a graph
    #   sequential_algorithm
    time_list = list()
    for i in range(k):
        random_deg_seq = graphical_random_generator(n)
        start = time.time()
        adj_dict_generated = algorithm(random_deg_seq)
        time_elapsed = time.time() - start
        time_list.append(time_elapsed)
    print("The algorithm", algorithm.__name__,"needs approximately",(sum(time_list)/k) ,"seconds to generate a graph.","(the average of",k,"trials.)")
    return(sum(time_list)/k)

#random.seed(2)
#measure_algorithm_k(10000,graph_generation_HH_random_to_high,1)
#measure_algorithm_k(10000,graph_generation_HH_highest_to_high,1)
#measure_algorithm_k(10000,graph_generation_HH_smallest_to_high,1)
#measure_algorithm_k(50,sequential_algorithm,10)


#------- Density Anlaysis  --------------------------------------------------------------------------------------------------

def find_density(degree_seq):
    #this function takes a degree sequence and returns the density of it
    m = int(sum(degree_seq)/2)
    n = len(degree_seq)
    return 2*m / (n*(n-1))

def density_disconnected_percent(a,b,algorithm,generator):
    #this function finds the percentage of disconnected graphs among all densities observed
    #a: number of graphs observed
    #b: lentgh of the degree sequence
    #choose the algorithm from these options:
    #   graph_generation_HH_random_to_high
    #   graph_generation_HH_highest_to_high
    #   graph_generation_HH_smallest_to_high
    #   graph_generation_HH_smallest_to_small #most of the time this algorithm cannot build a graph
    #   sequential_algorithm
    #choose the random sequence generator from these options:
    #   graphical_random_generator  
    #   graphical_exponential_generator
      
    m_n_counts = dict()
    m_n_disconnected = dict()
    m_n_percentages = dict()
    for x in range(a):                                          
        degree_seq = generator(b)         
        m_n = find_density(degree_seq)
        degree_matrix = algorithm(degree_seq)      
        
        if m_n in list(m_n_counts.keys()):
            m_n_counts[m_n] += 1
        else:
            m_n_counts[m_n] = 1
        
        if not breadth_first_search(degree_matrix):
            if m_n in list(m_n_disconnected.keys()):
                m_n_disconnected[m_n] += 1
            else:
                m_n_disconnected[m_n] = 1
    
    for key in list(m_n_counts.keys()):
        m_n_percentages[key] = 0

    for key in list(m_n_disconnected.keys()):
        m_n_percentages[key] = round(m_n_disconnected[key] / m_n_counts[key], 3)
        
    return sorted(m_n_percentages.items())

#random.seed(2)
#data = density_disconnected_percent(100000,5,graph_generation_HH_smallest_to_high,graphical_exponential_generator)

def plot_density_graph(data):
    #this function is used to visualize the disconnected percentage vs the m/n ratio  
    #x = [d[0] for d in data if d[1]!=0]         # use these x and y if you don't want 0 in the graph 
    #y = [d[1] for d in data if d[1]!=0]
    x = [d[0] for d in data]                   # use these x and y if you want 0 in the graph 
    y = [d[1] for d in data]

    plt.plot(x, y, '-o')
    plt.xlabel('density')
    plt.ylabel('disconnected percentage')
    plt.title('n=5, HH_smallest_to_high, disconnected percentage (exp) vs. density')
    plt.show()

#plot_density_graph(data)


#----------REPORT PREPERATION ----------------------------------------------------------------------------------------------------------

#---------- Input Format ---------------------------------------------------------------------------------------------------------------
def input_format(seed,n,input_id):
    group_id = "Group4"
    #this function is used to produce the appropriate input format
    random.seed(seed)                                       #set seed
    random_deg_seq = graphical_random_generator(n)          #generate a random degree sequence of length n
    m = int(sum(random_deg_seq)/2)                      
    n = len(random_deg_seq)
    filename = f"{group_id}-{n}-{m}-Input-{input_id}.txt"
    print(filename)
    print(*random_deg_seq)                                 #this will constitute the degree sequence
    return random_deg_seq

#Input_1 = input_format(2,10,1)     # Don't forget to change the input id


#---------- Output Format --------------------------------------------------------------------------------------------------------------

def output_format(input,algorithm,input_id,output_id):
    #this function is used to produce the appropriate input format
    #specify input
    #choose the algorithm from these options:
    #   graph_generation_HH_random_to_high
    #   graph_generation_HH_highest_to_high
    #   graph_generation_HH_smallest_to_high
    #   graph_generation_HH_smallest_to_small #most of the time this algorithm cannot build a graph
    #   sequential_algorithm
    group_id = "Group4"
    m = int(sum(input)/2)                      
    n = len(input)
    algorithm_name = algorithm.__name__
    filename = f"{group_id}-{n}-{m}-Input-{input_id}-Output-{output_id}-{algorithm_name}.txt"
    print(filename)
    degree_matrix_input = algorithm(input)
    connectivity = breadth_first_search(degree_matrix_input)
    print(int(connectivity))  
    visualize(degree_matrix_input)                                          # Whether the graph is connected or not (0 if not connected, 1 if connected)
    if connectivity:
        print(0)                                                        # If connected no need for pairwise edge interchange
    else:
        print(pairwise_edge_interchange(degree_matrix_input)[0])             # Number of Pairwise Interchanges

    print(measure_time(input,algorithm))                                # Time it takes to generate the graph 
    
    for key in list(degree_matrix_input.keys()):
        print(*degree_matrix_input[key])
    visualize(degree_matrix_input)
    
#output_format(Input_1,sequential_algorithm,1,1)       # Don't forget to change the input_id and output_id.