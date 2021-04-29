import json
import numpy as np


class Graph():
    def __init__(self, filename:str):
        with open(filename,'r') as f:
            self._graph = json.load(f)
            self._cost_graph = self._calculate_cost_dict()
            self._inf_cost = self._calculate_inf_cost_dict()
            self.edges_lines = self._calculate_egdes_dict()

    def _calculate_egdes_dict(self):
        egdes_lines = dict()
        for vertice in self._graph:
            egdes_lines[vertice] = dict()
            for neighbour in self._graph[vertice]['neighbors']:
                egdes_lines[vertice][neighbour] = np.linspace(self._graph[vertice]['xyz'], self._graph[neighbour]['xyz'], 50)
        return egdes_lines

    def _calculate_cost(self, A: str, B: str):
        return np.linalg.norm(np.array(self._graph[A]['xyz']) - np.array(self._graph[B]['xyz']))

    def _calculate_inf_cost_dict(self):
        inf_cost = dict()
        for vertice in self._graph:
            inf_cost[vertice] = np.inf
        return inf_cost

    def _calculate_cost_dict(self):
        cost_dict = dict()
        for vertice in self._graph:
            cost_dict[vertice] = dict()
            for neighbour in self._graph[vertice]['neighbors']:
                cost_dict[vertice][neighbour] = self._calculate_cost(
                    vertice, neighbour)
        return cost_dict

    def _search(self, source, target, can_neighborhood=True):
        parents = dict()
        costs = dict(self._inf_cost)

        cost_graph = dict(self._cost_graph)
        if can_neighborhood == False:
            cost_graph[source][target] = np.inf

        costs[source] = 0.0
        nextNode = source
        while nextNode != target:
            for neighbor in cost_graph[nextNode]:
                if cost_graph[nextNode][neighbor] + costs[nextNode] < costs[neighbor]:
                    costs[neighbor] = cost_graph[nextNode][neighbor] + \
                        costs[nextNode]
                    parents[neighbor] = nextNode
                del cost_graph[neighbor][nextNode]
            del costs[nextNode]
            nextNode = min(costs, key=costs.get)
        return parents

    def dijkstra_shortest(self, source, target, can_neighborhood=True):
        searchResult = self._search(source, target, can_neighborhood)
        node = target
        backpath = [target]
        path = []
        while node != source:
            backpath.append(searchResult[node])
            node = searchResult[node]
        for i in range(len(backpath)):
            path.append(backpath[-i - 1])
        return path
    
    def get_closest_vertice(self, xyz:tuple):
        distance = dict()
        for vertice in self._graph:
            distance[vertice] = np.linalg.norm(np.array(xyz) - np.array(self._graph[vertice]['xyz']), axis=-1)
        sort = sorted(distance, key=distance.get)
        return sort[0], distance[sort[0]]
    
    def get_closest_edges(self, xyz:tuple):
        closest = np.inf
        i = 1
        a, b = None, None
        for vertice1 in self.edges_lines:
            for vertice2 in self.edges_lines[vertice1]:
                dist = np.min(np.abs(np.linalg.norm(np.array(self.edges_lines[vertice1][vertice2]) - np.array(xyz), axis=-1)))
                if dist < closest:
                    closest = dist
                    a = vertice1
                    b = vertice2
        return a, b
        
    
    
