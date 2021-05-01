import json
import numpy as np
import matplotlib.pyplot as pypl

class Graph():
    def __init__(self, filename:str):
        with open(filename,'r') as f:
            self._graph = json.load(f)
            self._cost_graph = self._calculate_cost_dict()
            self._inf_cost = self._calculate_inf_cost_dict()
            self._edges_lines = self._calculate_egdes_dict()

    @staticmethod
    def calculate_distance(x:tuple, y:tuple):
        return np.linalg.norm(np.array(x) - np.array(y))
    def get_points_coord(self, point:str):
        return self._graph[point]['xyz']

    def _calculate_egdes_dict(self, graph:dict = None, num = 50):
        egdes_lines = dict()
        if graph is None: graph = self._graph
        for vertice in graph:
            egdes_lines[vertice] = dict()
            for neighbour in graph[vertice]['neighbors']:
                egdes_lines[vertice][neighbour] = np.linspace(graph[vertice]['xyz'], graph[neighbour]['xyz'], num)
        return egdes_lines

    def _calculate_cost(self, A: str, B: str, graph:dict = None):
        if graph is None: graph = self._graph
        return np.linalg.norm(np.array(graph[A]['xyz']) - np.array(graph[B]['xyz']))

    def _calculate_inf_cost_dict(self, graph:dict = None):
        inf_cost = dict()
        if graph is None: graph = self._graph
        for vertice in graph:
            inf_cost[vertice] = np.inf
        return inf_cost

    def _calculate_cost_dict(self, graph:dict = None):
        cost_dict = dict()
        if graph is None: graph = self._graph
        for vertice in graph:
            cost_dict[vertice] = dict()
            for neighbour in graph[vertice]['neighbors']:
                cost_dict[vertice][neighbour] = self._calculate_cost(
                    vertice, neighbour, graph)
        return cost_dict

    def _search(self, source, target, can_neighborhood=True, graph_data:tuple = None):
        parents = dict()
        inf_costs, cost_graph = None, None
        if graph_data is None:
            inf_costs = dict(self._inf_cost)
            cost_graph = dict(self._cost_graph)
        else:
            inf_costs, cost_graph = graph_data

        if can_neighborhood == False:
            cost_graph[source][target] = np.inf

        inf_costs[source] = 0.0
        nextNode = source
        while nextNode != target:
            for neighbor in cost_graph[nextNode]:
                if cost_graph[nextNode][neighbor] + inf_costs[nextNode] < inf_costs[neighbor]:
                    inf_costs[neighbor] = cost_graph[nextNode][neighbor] + \
                        inf_costs[nextNode]
                    parents[neighbor] = nextNode
                del cost_graph[neighbor][nextNode]
            del inf_costs[nextNode]
            nextNode = min(inf_costs, key=inf_costs.get)
        return parents

    def dijkstra_shortest(self, source, target, can_neighborhood=True, graph_data:tuple = None):
        searchResult = self._search(source, target, can_neighborhood, graph_data)
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
    
    def get_closest_edges(self, xyz:tuple, edges_lines:dict = None):
        if edges_lines is None: edges_lines = self._edges_lines
        closest = np.inf
        a, b = None, None
        for vertice1 in edges_lines:
            for vertice2 in edges_lines[vertice1]:
                dist = np.min(np.abs(np.linalg.norm(np.array(edges_lines[vertice1][vertice2]) - np.array(xyz), axis=-1)))
                if dist < closest:
                    closest = dist
                    a = vertice1
                    b = vertice2
        return a, b
    
    def get_closest_point_edges_road(self, xyz_source:tuple, xyz_target:tuple, source='source', target = 'target'):
        # Znajdź krawędź najbliższą dla danego punktu docelowego
        a, b = self.get_closest_edges(xyz_target)

        # Wstaw pomiędzy te krawędź nowy punkt docelowy
        graph:dict = dict(self._graph)
        if not (source in graph.keys() and target in graph.keys()):
            graph[target] = {'xyz': xyz_target, 'neighbors': [a, b]} 
            graph[a]['neighbors'].remove(b)           # a -x- b 
            graph[b]['neighbors'].remove(a)           # b -x- a
            graph[a]['neighbors'].append(target)      # a --- target
            graph[b]['neighbors'].append(target)      # target --- b 
                                                      # a --- target --- b
            
            # Przelicz linie dla nowego grafu
            edges_lines = self._calculate_egdes_dict(graph)
            c, d = self.get_closest_edges(xyz_source, edges_lines)

            # Wstaw pomiędzy te krawędzie nowy punkt startowy
            graph[source] = {'xyz': xyz_source, 'neighbors': [c, d]} 
            graph[c]['neighbors'].remove(d)           # c -x- d 
            graph[d]['neighbors'].remove(c)           # d -x- c
            graph[c]['neighbors'].append(source)      # c --- source
            graph[d]['neighbors'].append(source)      # source --- d
                                                      # c --- source --- d

        # Policz nowe wagi
        cost_graph = self._calculate_cost_dict(graph)
        inf_cost = self._calculate_inf_cost_dict(graph)
        
        # Znajdź nakrótszą ścieżkę między dwoma punktami (z zawracaniem)
        shortest = self.dijkstra_shortest(source, target, True, [inf_cost, cost_graph])

        #edges_lines = self._calculate_egdes_dict(graph)
        # points = []
        # for i in range(1, len(shortest)):
        #     a = shortest[i - 1]
        #     b = shortest[i]
        #     points.append(edges_lines[a][b])

        # points = np.array(points)

        return shortest, graph

if __name__ == '__main__':
    world = Graph('../world_map.json')

    image_map = np.zeros(())
    points, path = world.get_closest_point_edges_road([-60, 0, -183], [51, 0, -159], "J", "AN")
    x = points[:,:1]
    y = points[:,2:]
    print(path)
    print(world.dijkstra_shortest('J', 'AN'))
    pypl.plot(y, x)
    pypl.show()

    
