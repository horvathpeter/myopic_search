from numpy.core.fromnumeric import size

from igraph import *
import igraph
import numpy as np
import matplotlib.pyplot as plt


def plot_layout(layout_to_plot):
    """
    Function to see the geographical locations of the nodes - coordinates got by layout algorithm
    :param layout_to_plot:
    :return:
    """
    x_coords = []
    y_coords = []
    i = 0
    for coords in layout_to_plot.coords:
        print i, " ", coords
        x_coords.append(coords[0])
        y_coords.append(coords[1])
        i += 1

    plt.scatter(x_coords, y_coords)
    # uncomment to show the image
    plt.show()


def euclidean_distance(source, target):
    """
    Compute Euclidean distance between two nodes of the graph
    :param source: cartesian coordinates of the initial node (x, y)
    :param target: cartesian coordinates of the target node (x, y)
    :return: Euclidean distance
    """
    np_source = np.array([source[0], source[1]])
    np_target = np.array([target[0], target[1]])
    return np.linalg.norm(np_source - np_target)


def assign_edge_weights(graph, graph_layout):
    """
    Assign weights to every edge of the graph. The weight is Euclidean distance between source and target of the edge.
    :param graph:
    :param graph_layout:
    :return:
    """
    for edge in graph.es:
        # print edge.source, edge.target
        dist = euclidean_distance(graph_layout[edge.source], graph_layout[edge.target])
        edge["weight"] = dist


def get_myopic_path(graph, graph_layout, source, target):
    closest = ""
    path = []

    # if graph.neighbors(source)[0] == target:
    if target in graph.neighbors(source):
        path = target
    else:
        temp_target = source

        visited = []
        visited.append(temp_target)
        # print "Visited list is ", visited
        # print "Setting temp target to ", temp_target

        while temp_target != target:

            for neighbour in graph.neighbors(temp_target):
                if neighbour not in visited:
                    closest = neighbour
                    break

            # print "Temp target is ",temp_target, "and setting closest to ",closest
            possibilities = 0
            for nghb in graph.neighbors(temp_target):
                # print "Distance of nghb", nghb, "to target is ", euclidean_distance(graph_layout[target],
                #                                                                     graph_layout[nghb])
                # print "checking if ", nghb, " is in list of visited which is ", visited, "and if ", euclidean_distance(
                #         graph_layout[target], graph_layout[nghb]), "is lower than ", euclidean_distance(
                #         graph_layout[target],
                #         graph_layout[closest])
                if nghb not in visited and (
                            euclidean_distance(graph_layout[target], graph_layout[nghb]) < euclidean_distance(
                                graph_layout[target],
                                graph_layout[closest])):
                    closest = nghb

                    # temp_target = nghb

            for pos in graph.neighbors(temp_target):
                if pos not in visited:
                    possibilities += 1

            if possibilities > 0:
                temp_target = closest
            else:
                temp_target = graph.neighbors(temp_target)[0]

            visited.append(temp_target)
            path.append(temp_target)

    # print closest
    return path


graf = igraph.Graph.Barabasi(n=1000, m=3)
#graf = igraph.Graph.Forest_Fire(n=300, fw_prob = 0.5)
#graf = Graph.Read_Pickle("barabasi_400_3")
#graf = Graph.Read_Pickle("forest_fire_1000")
#igraph.Graph.write_pickle(graf, "forest_fire_1000", version=1)
#igraph.Graph.write_pickle(graf, "barabasi_400_3", version=1)
print graf

# plot(graf, vertex_label=range(0, graf.vcount()))

#g_layout = graf.layout_circle()
g_layout = graf.layout_auto()
#g_layout =  graf.layout_reingold_tilford()

#plot_layout(g_layout)
assign_edge_weights(graf, g_layout)

#
# for edge in graf.es:
#     print edge.source, edge.target, edge["weight"]

# compute closenes centralities of the graph

#centralities = graf.closeness(weights="weight")
centralities = graf.eigenvector_centrality(weights="weight")

# build a list of tuples, one tuple = (node_id, node_centrality)
cent_ids = []
c = 0
for cent in centralities:
    cent_ids.append((c, cent))
    c += 1

# print cent_ids

# sort the list of tuples by the centrality from the biggest to lowest
sorted_centralities = sorted(cent_ids, key=lambda node_centrality: node_centrality[1], reverse=True)
print sorted_centralities

# take the first quarter of the sorted list - quarter of the most important nodes
first_quarter = sorted_centralities[0:graf.vcount() / 4]
three_quarters = sorted_centralities[graf.vcount() / 4: graf.vcount()]

# print size(first_quarter)
# print size(three_quarters)

# list of IDs of the first quarter
ids_of_first_quarter = []
ids_of_three_quarters = []

for record in first_quarter:
    ids_of_first_quarter.append(record[0])

for record in three_quarters:
    ids_of_three_quarters.append(record[0])

print ids_of_first_quarter
print ids_of_three_quarters

count = 0
for centrality in first_quarter:
    print count, centrality
    count += 1

order = 1
same_same = 0
same_same_but_different = 0

for record in first_quarter:
    print record[0]
    for vertex in range(order, size(ids_of_first_quarter)):
        # print "Getting myopic search from ", record[0], " to ", ids_of_first_quarter[vertex]
        myopic_path = get_myopic_path(graf, g_layout, record[0], ids_of_first_quarter[vertex])
        shortest_path = graf.get_shortest_paths(v=record[0], to=ids_of_first_quarter[vertex], weights="weight")
        # dijsktra_path = graf.shortest_paths_dijkstra(source=record[0], target=ids_of_first_quarter[vertex],
        #                                              weights="weight")

        # print "Myopic shortest path is ", myopic_path, " and size is ", size(myopic_path)
        # print "IGraph shortest path is ", shortest_path, " and size is ", size(shortest_path) - 1
        # print "Dijkst shortest path is ", dijsktra_path

        if size(myopic_path) == size(shortest_path) - 1:
            same_same += 1
        else:
            same_same_but_different += 1
    order += 1
#
# for record in first_quarter:
#     print record[0]
#     for vertex in ids_of_three_quarters:
#         # print "Getting myopic search from ", record[0], " to ", vertex
#         myopic_path = get_myopic_path(graf, g_layout, record[0], vertex)
#         shortest_path = graf.get_shortest_paths(v=record[0], to=vertex, weights="weight")
#         # dijkstra_path = graf.shortest_paths_dijkstra(source=record[0], target=vertex, weights="weight")
#
#         # print "Myopic shortest path is ", myopic_path, " ans size is ", size(myopic_path)
#         # print "IGraph shortest path is ", shortest_path, " and size is ", size(shortest_path) - 1
#         # print "Dijkst shortest path is ", dijkstra_path
#
#         if size(myopic_path) == size(shortest_path) - 1:
#             same_same += 1
#         else:
#             same_same_but_different += 1

# for record in three_quarters:
#     print record[0]
#     for vertex in range(order, size(ids_of_three_quarters)):
#         #print "Getting myopic search from ", record[0], " to ", ids_of_three_quarters[vertex]
#         myopic_path = get_myopic_path(graf, g_layout, record[0], ids_of_three_quarters[vertex])
#         shortest_path = graf.get_shortest_paths(v=record[0], to=ids_of_three_quarters[vertex], weights="weight")
#         # dijsktra_path = graf.shortest_paths_dijkstra(source=record[0], target=ids_of_first_quarter[vertex],
#         #                                              weights="weight")
#
#         # print "Myopic shortest path is ", myopic_path, " and size is ", size(myopic_path)
#         # print "IGraph shortest path is ", shortest_path, " and size is ", size(shortest_path) - 1
#         # print "Dijkst shortest path is ", dijsktra_path
#
#         if size(myopic_path) == size(shortest_path) - 1:
#             same_same += 1
#         else:
#             same_same_but_different += 1
#     order += 1



print "Result was same same ", same_same, " times"
print "Result was same same but different ", same_same_but_different, " times"
