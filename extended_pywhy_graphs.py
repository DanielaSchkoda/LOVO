import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import itertools as it
from pywhy_graphs import ADMG, PAG, CPDAG, mag_to_pag, valid_mag
from typing import Optional, Iterator
import causaldag
import random
from dodiscover.metrics import structure_hamming_dist as shd

class ExtendedADMG(ADMG):
    def __init__(
        self, 
        incoming_nodes=set(),
        incoming_directed_edges=None, 
        incoming_bidirected_edges=None, 
        incoming_undirected_edges=None, 
        **attr
    ):
        super().__init__(
            incoming_directed_edges=incoming_directed_edges, 
            incoming_bidirected_edges=incoming_bidirected_edges, 
            incoming_undirected_edges=incoming_undirected_edges,
            directed_edge_name='directed', 
            bidirected_edge_name='bidirected', 
            undirected_edge_name='undirected', 
            **attr
        )

        # ADMG already adds all nodes from the edges, add potential isolated nodes here.
        self.add_nodes_from(incoming_nodes)

    @classmethod
    def from_adjacency_matrix(cls, adj_matrix, RCD_matrix = False):
        '''
        Construct an ADMG corresponding to an adjacency matrix. 
        If RCD_matrix is True, the adjacency matrix is assumed to encode bidirected edges as NaNs.
        Otherwise, the left square part encodes the directed edges among observed nodes,
        and the remaining columns encode explicit latents, which lead to bidirected edges among their children.
        '''
        k = adj_matrix.shape[0]

        if RCD_matrix:
            nodes = set(range(k))
            directed_edges = [(i, j) for (i, j) in np.argwhere((adj_matrix.T != 0) & ~np.isnan(adj_matrix.T))]
            # Nans encode bidirected edges
            bidirected_edges = [(i, j) for (i, j) in np.argwhere(np.isnan(adj_matrix.T)) if i < j]
            return cls(nodes, directed_edges, bidirected_edges)
        else:
            observed_part, unobserved_part = adj_matrix[:, :k], adj_matrix[:, k:]
            nodes = set(range(k))
            directed_edges = observed_part.T 
            # Remove edge weights
            directed_edges[directed_edges != 0] = 1
            bidirected_edges = []
            for l in range(unobserved_part.shape[1]):
                children_l = np.argwhere(unobserved_part[:, l] != 0).flatten()
                bidirected_edges.extend(it.combinations(children_l, 2))
            return cls(nodes, directed_edges, bidirected_edges)
    
    def siblings(self, X) -> Iterator:
        '''
        Returns the siblings of X.
        '''
        for nbr in self.neighbors(X):
            if (
                self.has_edge(nbr, X, self.bidirected_edge_name)
            ):
                yield nbr
    
    def project_to_GX(self, X):
        '''
        Project the graph onto the nodes without the node X using the marginalization rules described in the paper.
        '''
        G_X = ExtendedADMG(incoming_nodes = self.nodes - {X}, 
            incoming_directed_edges = [e for e in self.directed_edges if X not in e],
            incoming_bidirected_edges = [e for e in self.bidirected_edges if X not in e])
        for c in self.children(X):
            for p in self.parents(X):
                G_X.add_edge(p, c, 'directed') 
            for c2 in self.children(X):
                if c2 != c:
                    G_X.add_edge(c, c2, 'bidirected')
            for s in self.siblings(X):
                if s != c:
                    G_X.add_edge(c, s, 'bidirected')
        return G_X
    
    def to_CPDAG(self) -> CPDAG:
        G_oriented = causaldag.DAG(arcs=self.directed_edges)

        # Orient bidirected edges randomly to create a DAG.
        # In doing so ensure no cycle is created
        for (u, v) in self.bidirected_edges:
            if G_oriented.is_ancestor_of(u, v):
                G_oriented.add_arc(u, v)
            elif G_oriented.is_ancestor_of(v, u):
                G_oriented.add_arc(v, u)
            elif random.choice([True, False]):
                G_oriented.add_arc(u, v)
            else:
                G_oriented.add_arc(v, u)

        # Use package causaldag to create CPDAG and then revert back to phywhy class
        cpdag = G_oriented.cpdag()
        cpdag = CPDAG(cpdag.arcs,
                        cpdag.edges)
        
        # Add all nodes to ensure isolated nodes are included
        cpdag.add_nodes_from(self.nodes)
        return cpdag
    
    def to_RCD_ADMG(self) -> ADMG:
        return ExtendedADMG(
            incoming_nodes=self.nodes,
            incoming_directed_edges=self.directed_edges,
            incoming_bidirected_edges=[(i, j) for (i, j) in self.bidirected_edges if not (i, j) in self.directed_edges or (j, i) in self.directed_edges]
        )
    
    def to_PAG(self) -> Optional[PAG]:
        if valid_mag(self):
            try:
                pag = mag_to_pag(self)
            except:
                print('Warning: mag_to_pag failed')
                return None
            return ExtendedPAG(
                    incoming_nodes=self.nodes,
                    incoming_directed_edges=pag.directed_edges,
                    incoming_bidirected_edges=pag.bidirected_edges,
                    incoming_circle_edges=pag.circle_edges,
                    incoming_undirected_edges=pag.undirected_edges
                    )
        else:
            return None

    def plot(self):
        G_directed = self.get_graphs('directed')
        G_bidirected = self.get_graphs('bidirected').to_directed()

        pos = nx.spring_layout(G_directed)
        nx.draw_networkx_nodes(G_directed, pos, node_color='grey', node_size=300)
        nx.draw_networkx_labels(G_directed, pos, font_size=10, font_weight='bold')

        nx.draw_networkx_edges(G_directed, pos, edge_color='blue', arrows=True, arrowsize=15)
        bidirected_edges = G_bidirected.edges()
        first_half = [(i, j) for i, j in bidirected_edges if i < j]
        reversed_edges = [(i, j) for i, j in bidirected_edges if i > j]
        nx.draw_networkx_edges(G_bidirected, pos, edgelist=first_half, edge_color='lightblue', connectionstyle='arc3,rad=0.2', style='dashed', arrows=True, arrowsize=15)
        nx.draw_networkx_edges(G_bidirected, pos, edgelist=reversed_edges, edge_color='lightblue', connectionstyle='arc3,rad=-0.2', style='dashed', arrows=True, arrowsize=15)

        plt.show()

class ExtendedPAG(PAG):
    def __init__(
        self,
        incoming_nodes=set(),
        incoming_directed_edges=None,
        incoming_undirected_edges=None,
        incoming_bidirected_edges=None,
        incoming_circle_edges=None,
        **attr,
    ):
        super().__init__(
            incoming_directed_edges=incoming_directed_edges,
            incoming_undirected_edges=incoming_undirected_edges,
            incoming_bidirected_edges=incoming_bidirected_edges,
            incoming_circle_edges=incoming_circle_edges,
            directed_edge_name="directed",
            undirected_edge_name="undirected",
            bidirected_edge_name="bidirected",
            circle_edge_name="circle",
            **attr,
        )
        
        # super().__init__ already adds all nodes from the edges, add potential isolated nodes here.
        self.add_nodes_from(incoming_nodes)

    def possible_siblings(self, X) -> Iterator:
        '''
        Returns the set of nodes that could be siblings of X.
        '''
        for nbr in self.neighbors(X):
            if (
                self.has_edge(nbr, X, self.bidirected_edge_name)
                or self.has_edge(nbr, X, self.circle_edge_name)
                or self.has_edge(X, nbr, self.circle_edge_name)
            ):
                yield nbr

def combine_graphs(G_X, G_Y):
    '''
    Combines two graphs (any graph type that has directed, bidirectedm, circle and/or undirected edges) into one PAG.
    '''
    # Prioritize directed and bidirected edges
    G = ExtendedPAG()
    relevant_edge_types = [type for type in ['directed', 'bidirected', 'circle', 'undirected'] if type in G_X.edge_types]
    for type in relevant_edge_types:
        new_edges = set(G_X.edges()[type]) | set(G_Y.edges()[type])
        existing_edges = {existing_edge for edges in G.edges().values() for existing_edge in edges} | {(u, v) for (v, u) in G.bidirected_edges()}
        # Bidirected edges and undirected can only be added if neither the edge nor the edge reversed is already in the graph
        if type in ['bidirected', 'undirected'] and isinstance(G_X, PAG):
            existing_edges_reversed = {tuple(reversed(existing_edge)) for existing_edge in existing_edges}
            G.add_edges_from(new_edges - existing_edges - existing_edges_reversed, type)
        elif type == 'undirected' and isinstance(G_X, CPDAG):
            existing_edges_reversed = {tuple(reversed(existing_edge)) for existing_edge in existing_edges}
            new_edges_reversed = {tuple(reversed(new_edge)) for new_edge in new_edges}
            G.add_edges_from((new_edges | new_edges_reversed) - (existing_edges | existing_edges_reversed), 'circle')
        else:
            G.add_edges_from(new_edges - existing_edges, type)
    
    return G

def find_separating_set_PAG(X, Y, G: ExtendedPAG):
    nx_G = to_nx_graph(G)
    if X not in nx_G.nodes or Y not in nx_G.nodes:
        return []
    paths = list(nx.all_simple_paths(nx_G, X, Y))
    short_paths = [path for path in paths if len(path) == 3]
    for path in short_paths:
        middle_node = path[1]
        if middle_node in G.possible_children(X) and middle_node in G.possible_children(Y):
            return None
    return list({node for path in paths for node in path} - {X, Y})

def to_nx_graph(G):
    '''
    Converts a PAG to an undirected NetworkX graph.
    '''
    nx_G = nx.Graph()
    for edges_of_this_type in G.edges().values():
        nx_G.add_edges_from(edges_of_this_type)
    return nx_G

def calculate_shd(G_hat, G_true):
    '''
    Calculate the structural Hamming distance between two graphs. For PAGs, there is no straightforward way to calculate the distance.
    Therefore, the method simply returns the SHD of the skeletons.
    '''
    if isinstance(G_hat, ADMG):
        shd_directed = shd(G_hat.sub_directed_graph(), G_true.sub_directed_graph())
        shd_bidirected = shd(G_hat.sub_bidirected_graph(), G_true.sub_bidirected_graph(), double_for_anticausal=False)
        return shd_directed + shd_bidirected
    elif isinstance(G_hat, CPDAG):
        shd_directed = shd(G_hat.sub_directed_graph(), G_true.sub_directed_graph())
        shd_undirected = shd(G_hat.sub_undirected_graph(), G_true.sub_undirected_graph(), double_for_anticausal=False)
        return shd_directed + shd_undirected
    elif isinstance(G_hat, PAG):
        return shd(G_hat.to_undirected(), G_true.to_undirected(), double_for_anticausal=False)
    else:
        raise ValueError(f'Unsupported graph type: {type(G_hat), isinstance(G_hat, ADMG)}')


