import pandas as pd
import itertools
import networkx as nx
from itertools import combinations
import spacy
from nltk.stem import SnowballStemmer
import string
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import warnings
import scipy.sparse.linalg
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

from text_func import wordopt, process_text


def create_graph_from_dataframe(dataframe, token_column):

    G = nx.Graph()

    for tokens in dataframe[token_column]:

        for i in range(len(tokens) - 1):
            word1, word2 = tokens[i], tokens[i + 1]
            if G.has_edge(word1, word2):
                G[word1][word2]['weight'] += 1
            else:
                G.add_edge(word1, word2, weight=1)

        for word1, word2 in itertools.combinations(tokens, 2):

            if abs(tokens.index(word1) - tokens.index(word2)) > 1:
                if G.has_edge(word1, word2):
                    G[word1][word2]['weight'] += 0.5
                else:
                    G.add_edge(word1, word2, weight=0.5)

    return G

def charac_graph(G):
    order = G.order()
    print("Rząd - liczba wierzchołków: ", order)

    size = G.size()
    print("Rozmiar - liczba krawędzi: ", size)

    density = nx.density(G)
    print("Gęstość grafu: ", density)

    if nx.is_connected(G):
        diameter = nx.diameter(G)
        print("Średnica grafu: ", diameter)
    else:
        print("Graf nie jest spójny - Brak średnicy ")

    if nx.is_connected(G):
        avg_shortest_path_length = nx.average_shortest_path_length(G)
        print("Średnia długość ścieżki: ", avg_shortest_path_length)
    else:
        print("Graf nie jest spójny - Brak średniej długości ścieżki ")


def charac_node(G):
    # Stopień wierzchołka
    degree_dict = dict(G.degree())
    print("Stopień wierzchołka dla wierzchołka: ")

    sorted_data = sorted(degree_dict.items(), key=lambda x: x[1], reverse=True)
    for node, degree in sorted_data[:3]:
        print(f"Węzeł {node}: Stopień {degree} ")

    avg_degree = sum(degree_dict.values()) / G.order()
    print("\nŚredni stopień węzła: ", avg_degree)

    closeness_dict = nx.closeness_centrality(G)
    print("\nBliskość dla wierzchołka: ")

    sorted_data = sorted(closeness_dict.items(), key=lambda x: x[1], reverse=True)
    for node, closeness in sorted_data[:3]:
        print(f"Węzeł {node}: Bliskość {closeness:.3f} ")

    betweenness_dict = nx.betweenness_centrality(G)
    print("\nPośrednictwo dla każdego wierzchołka: ")

    sorted_data = sorted(betweenness_dict.items(), key=lambda x: x[1], reverse=True)
    for node, betweenness in sorted_data[:3]:
        print(f"Węzeł {node}: Pośrednictwo {betweenness:.3f}")


def charac_edge(G):
    edge_betweenness = nx.edge_betweenness_centrality(G)

    print("\n\nPośrednictwo krawędzi\n ")

    sorted_data = sorted(edge_betweenness.items(), key=lambda x: x[1], reverse=True)
    for edge, numb in sorted_data[:3]:
        print(f"Krawędź {edge}: Pośrednictwo {numb:.3f} ")

    print("\n\nWagi\n")

    edge_weights = {}
    for u, v, data in G.edges(data=True):
        edge_weights[(u, v)] = data['weight']

    sorted_data = sorted(edge_weights.items(), key=lambda x: x[1], reverse=True)
    for edge, numb in sorted_data[:3]:
        print(f"Krawędź {edge}: Waga {numb:.3f}")

    print("\n\nCentralność wektora własnego\n ")

    eigenvector_centrality = nx.eigenvector_centrality(G, weight='weight')

    sorted_data = sorted(eigenvector_centrality.items(), key=lambda x: x[1], reverse=True)
    for node, numb in sorted_data[:3]:
        print(f"Wierzchołek {node}: Centralność wektora własnego {numb:.3f} ")

    print("\n\nPageRank dla wierzchołka\n ")

    pagerank = nx.pagerank(G, weight='weight')

    sorted_data = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)
    for node, numb in sorted_data[:3]:
        print(f"Wierzchołek {node}: PageRank {numb:.3f} ")

    print("\n\nPageRank dla krawędzi\n ")

    edge_pagerank = {}
    for u, v in G.edges():
        edge_pagerank[(u, v)] = pagerank[u] + pagerank[v]

    sorted_data = sorted(edge_pagerank.items(), key=lambda x: x[1], reverse=True)
    for node, numb in sorted_data[:3]:
        print(f"Krawędź {node}: PageRank {numb:.3f}")

def calculate_vertex_connectivity(G):

    n = len(G.nodes())
    for k in range(1, n):
        for nodes in combinations(G.nodes(), k):
            G_copy = G.copy()
            G_copy.remove_nodes_from(nodes)
            if nx.is_connected(G_copy) is False:
                return k
    return n
def analyze_graph(G):

    connected_components = list(nx.connected_components(G))
    print("Składowe spójne:")
    for i, component in enumerate(connected_components):
        print(f"Składowa {i + 1}: {component}")

    vertex_connectivity_value = calculate_vertex_connectivity(G)
    print("\nK-spójność wierzchołkowa: ", vertex_connectivity_value)

    clique = nx.approximation.max_clique(G)
    print("\nKlika maksymalna: ", clique)

    bridges = list(nx.bridges(G))
    print("\nMosty (krawędzie krytyczne): ")
    if bridges:
        for bridge in bridges:
            print(f"Most: {bridge} ")
    else:
        print("Brak mostów w grafie ")

    articulation_points = list(nx.articulation_points(G))
    print("\nPrzeguby (punkty artykulacji): ")
    if articulation_points:
        for point in articulation_points:
            print(f"Przegub: {point} ")
    else:
        print("Brak przegubów w grafie ")